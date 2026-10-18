# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import Counter

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from trl import GRPOConfig, GRPOTrainer, RLOOConfig, RLOOTrainer

from .testing_utils import TrlTestCase


def _tiny_model_and_tokenizer(num_prompts, path):
    # Built locally: only the prompt schedule is under test, not the weights.
    words = ["<pad>", "<eos>", "<unk>"] + [f"p{i}" for i in range(num_prompts)] + list("abcdefgh")
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.padding_side = "left"
    config = Qwen2Config(
        vocab_size=len(words),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=1,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    # RLOOTrainer builds its reference model from `model.config._name_or_path`, so reload from disk.
    Qwen2ForCausalLM(config).save_pretrained(path)
    return Qwen2ForCausalLM.from_pretrained(path), tokenizer


class TestResumeSchedule(TrlTestCase):
    @pytest.mark.parametrize("trainer_cls, config_cls", [(GRPOTrainer, GRPOConfig), (RLOOTrainer, RLOOConfig)])
    @pytest.mark.parametrize(
        "num_iterations, steps_per_generation, grad_accum, num_prompts, epochs, resume_step, ignore_data_skip",
        [
            (2, 1, 1, 24, 1, 1, False),  # checkpoint inside a 2-update block
            (3, 1, 2, 24, 1, 4, False),  # 3-update block, gradient accumulation
            (2, 1, 1, 24, 1, 3, False),  # last update of a block
            (2, 1, 1, 24, 1, 1, True),  # no batch skipping: epoch restarts, so a new block starts
        ],
    )
    def test_resume_keeps_prompt_schedule(
        self,
        trainer_cls,
        config_cls,
        num_iterations,
        steps_per_generation,
        grad_accum,
        num_prompts,
        epochs,
        resume_step,
        ignore_data_skip,
    ):
        # After a resume, each optimizer update must train on the same prompts as in the uninterrupted run.
        # Completions of the interrupted block are regenerated, so only prompts are compared.
        def run(output_dir, resume_from_checkpoint=None):
            model, tokenizer = _tiny_model_and_tokenizer(num_prompts, os.path.join(self.tmp_dir, "base"))
            dataset = Dataset.from_dict({"prompt": [f"p{i} a b" for i in range(num_prompts)]})
            args = config_cls(
                output_dir=output_dir,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=grad_accum,
                num_generations=2,
                num_iterations=num_iterations,
                steps_per_generation=steps_per_generation,
                max_completion_length=4,
                num_train_epochs=epochs,
                save_steps=1,
                save_only_model=False,
                report_to="none",
                use_cpu=True,
                seed=123,
                data_seed=123,
                ignore_data_skip=ignore_data_skip,
                shuffle_dataset=not ignore_data_skip,
            )
            schedule = {}

            class RecordingTrainer(trainer_cls):
                def compute_loss(self, model, inputs, *a, **kw):
                    if model.training:
                        prompts = self.processing_class.batch_decode(inputs["prompt_ids"], skip_special_tokens=True)
                        schedule.setdefault(self.state.global_step, Counter()).update(prompts)
                    return super().compute_loss(model, inputs, *a, **kw)

            trainer = RecordingTrainer(
                model=model,
                reward_funcs=lambda completions, **kwargs: [float(len(c) % 3) for c in completions],
                args=args,
                train_dataset=dataset,
                processing_class=tokenizer,
            )
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            return schedule

        full_dir = os.path.join(self.tmp_dir, "full")
        full = run(full_dir)
        resumed = run(os.path.join(self.tmp_dir, "resumed"), os.path.join(full_dir, f"checkpoint-{resume_step}"))
        assert min(resumed) == resume_step
        if ignore_data_skip:
            # The Trainer restarts the epoch, so the resumed run replays the schedule from its first block.
            for step, prompts in resumed.items():
                assert prompts == full[step - resume_step], f"update {step} did not follow a restarted epoch"
            return
        for step, prompts in resumed.items():
            assert prompts == full[step], f"update {step} trained on different prompts after resume"
