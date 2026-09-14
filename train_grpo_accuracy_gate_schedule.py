import argparse
import copy
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from tqdm.auto import tqdm
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import TrainerCallback
from trl import GRPOConfig

from dataset_utils import build_grpo_dataset
from instrumented_grpo_trainer import InstrumentedGRPOTrainer
from model_utils import load_model_for_training, load_tokenizer, resolve_cached_model_path
from reward_utils import CompletionTokenLengthReward, final_answer_format_reward, gsm8k_correctness_reward


CHECKPOINT_STEP_RE = re.compile(r"checkpoint-(\d+)$")


@dataclass(frozen=True)
class AccuracyGateStage:
    min_correct_rate: float
    max_correct_rate: float
    update_steps: int
    start_step: int
    end_step: int


def parse_stage_schedule(schedule_text: str) -> list[AccuracyGateStage]:
    if not schedule_text:
        raise ValueError("--stage_schedule must not be empty.")

    stages = []
    cumulative_steps = 0
    for raw_item in schedule_text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "Each stage must use RANGE:STEPS format, for example "
                "'0.875-0.625:50,0.75-0.5:50'."
            )

        range_text, steps_text = item.rsplit(":", 1)
        if "-" in range_text:
            left, right = range_text.split("-", 1)
        elif "~" in range_text:
            left, right = range_text.split("~", 1)
        else:
            raise ValueError("Stage range must use A-B or A~B format.")

        first_rate = float(left.strip())
        second_rate = float(right.strip())
        min_rate = min(first_rate, second_rate)
        max_rate = max(first_rate, second_rate)
        update_steps = int(steps_text.strip())
        if not 0.0 <= min_rate <= 1.0 or not 0.0 <= max_rate <= 1.0:
            raise ValueError(f"Stage range must be within [0, 1], got {range_text}.")
        if update_steps <= 0:
            raise ValueError(f"Stage update steps must be positive, got {steps_text}.")

        start_step = cumulative_steps
        cumulative_steps += update_steps
        stages.append(
            AccuracyGateStage(
                min_correct_rate=min_rate,
                max_correct_rate=max_rate,
                update_steps=update_steps,
                start_step=start_step,
                end_step=cumulative_steps,
            )
        )

    if not stages:
        raise ValueError("--stage_schedule must contain at least one stage.")
    return stages


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_step(path: Optional[str]) -> int:
    if not path:
        return 0
    match = CHECKPOINT_STEP_RE.search(Path(path).name)
    return int(match.group(1)) if match else 0


def checkpoint_trainer_global_step(path: Optional[str]) -> int:
    if not path:
        return 0
    state_path = Path(path) / "trainer_state.json"
    if not state_path.exists():
        return 0
    try:
        with state_path.open("r", encoding="utf-8") as state_file:
            state = json.load(state_file)
    except (OSError, json.JSONDecodeError):
        return 0
    return int(state.get("global_step", 0) or 0)


class StopAfterAcceptedUpdatesCallback(TrainerCallback):
    def __init__(self, trainer_ref, target_update_steps: int):
        self.trainer_ref = trainer_ref
        self.target_update_steps = target_update_steps

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer_ref()
        if trainer is not None and trainer.accepted_update_steps >= self.target_update_steps:
            control.should_training_stop = True
        return control


class AcceptedUpdateProgressAndCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        trainer_ref,
        target_update_steps: int,
        save_steps: int,
        initial_accepted_update_steps: int,
    ):
        self.trainer_ref = trainer_ref
        self.target_update_steps = target_update_steps
        self.save_steps = save_steps
        self.initial_accepted_update_steps = initial_accepted_update_steps
        self.last_progress_update = initial_accepted_update_steps
        self.last_checkpoint_update = initial_accepted_update_steps
        self.progress_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.trainer_ref()
        if trainer is not None:
            trainer.accepted_update_steps = max(
                trainer.accepted_update_steps,
                self.initial_accepted_update_steps,
            )
            self.last_progress_update = trainer.accepted_update_steps
            self.last_checkpoint_update = trainer.accepted_update_steps

        if state.is_local_process_zero:
            self.progress_bar = tqdm(
                total=self.target_update_steps,
                initial=min(self.last_progress_update, self.target_update_steps),
                desc="Accepted updates",
                dynamic_ncols=True,
            )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer_ref()
        if trainer is None:
            return control

        accepted_steps = trainer.accepted_update_steps
        if self.progress_bar is not None:
            progress_steps = min(accepted_steps, self.target_update_steps)
            delta = progress_steps - min(self.last_progress_update, self.target_update_steps)
            if delta > 0:
                self.progress_bar.update(delta)
            self.progress_bar.set_postfix(
                attempt_step=state.global_step,
                rejected=trainer.rejected_update_steps,
                refresh=False,
            )
            self.last_progress_update = accepted_steps

        if self.save_steps > 0 and accepted_steps - self.last_checkpoint_update >= self.save_steps:
            self._save_accepted_update_checkpoint(trainer, accepted_steps)
            self.last_checkpoint_update = accepted_steps

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None
        return control

    def _save_accepted_update_checkpoint(self, trainer, accepted_steps: int):
        output_dir = os.path.join(trainer.args.output_dir, f"checkpoint-{accepted_steps}")
        if trainer.args.should_save:
            print(
                f"\n保存有效更新 checkpoint: {output_dir} "
                f"(accepted_update_steps={accepted_steps}, attempt_step={trainer.state.global_step})"
            )

        trainer.save_model(output_dir, _internal_call=True)
        if not trainer.args.save_only_model:
            trainer._save_optimizer_and_scheduler(output_dir)
            trainer._save_scaler(output_dir)
            trainer._save_rng_state(output_dir)
        if trainer.args.should_save:
            trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))


class AccuracyGateGRPOTrainer(InstrumentedGRPOTrainer):
    def __init__(
        self,
        *args,
        min_correct_rate: float,
        max_correct_rate: float,
        accepted_length_min: int = 0,
        accepted_length_max: Optional[int] = None,
        rejected_length_only_update: bool = False,
        rejected_length_only_advantage_scale: float = 1.0,
        max_retry_truncated_count: int = 0,
        max_truncation_retry_rounds: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_correct_rate = min_correct_rate
        self.max_correct_rate = max_correct_rate
        self.accepted_length_min = accepted_length_min
        self.accepted_length_max = accepted_length_max
        self.rejected_length_only_update = rejected_length_only_update
        self.rejected_length_only_advantage_scale = rejected_length_only_advantage_scale
        self.max_retry_truncated_count = max_retry_truncated_count
        self.max_truncation_retry_rounds = max_truncation_retry_rounds
        self.accepted_update_steps = 0
        self.rejected_update_steps = 0
        self._last_train_step_has_accepted_group = False

    def _is_completion_truncated_ids(self, ids) -> bool:
        if not ids:
            return False
        eos_token_id = getattr(self, "eos_token_id", None)
        eos_token_ids = set()
        if eos_token_id is not None:
            if isinstance(eos_token_id, (list, tuple, set)):
                eos_token_ids.update(int(token_id) for token_id in eos_token_id)
            else:
                eos_token_ids.add(int(eos_token_id))

        pad_token_id = getattr(self, "pad_token_id", None)
        if pad_token_id is not None:
            eos_token_ids.add(int(pad_token_id))

        return len(ids) >= self.args.max_completion_length and int(ids[-1]) not in eos_token_ids

    def _set_retry_generation_count(self, count: int):
        old_num_generations = self.num_generations
        old_generation_num_return_sequences = getattr(self.generation_config, "num_return_sequences", None)
        old_generation_num_generations = getattr(self.generation_config, "num_generations", None)

        self.num_generations = count
        if old_generation_num_return_sequences is not None:
            self.generation_config.num_return_sequences = count
        if old_generation_num_generations is not None:
            self.generation_config.num_generations = count

        def restore():
            self.num_generations = old_num_generations
            if old_generation_num_return_sequences is not None:
                self.generation_config.num_return_sequences = old_generation_num_return_sequences
            if old_generation_num_generations is not None:
                self.generation_config.num_generations = old_generation_num_generations

        return restore

    def _generate_retry_batch(self, prompts, retry_count: int):
        restore = self._set_retry_generation_count(retry_count)
        mode = "train" if self.model.training else "eval"
        metric_lengths = {key: len(values) for key, values in self._metrics[mode].items()}
        old_num_input_tokens_seen = self.state.num_input_tokens_seen
        try:
            return super()._generate(prompts)
        finally:
            self.state.num_input_tokens_seen = old_num_input_tokens_seen
            for key, old_len in metric_lengths.items():
                if key in self._metrics[mode]:
                    del self._metrics[mode][key][old_len:]
            restore()

    def _retry_truncated_completions(self, prompts, prompt_ids, completion_ids, logprobs, extra_fields):
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        if (
            mode != "train"
            or self.max_retry_truncated_count <= 0
            or self.max_truncation_retry_rounds <= 0
            or num_generations <= 0
            or len(completion_ids) % num_generations != 0
            or logprobs is not None
            or getattr(self, "tools", None)
        ):
            return prompt_ids, completion_ids, logprobs, extra_fields

        group_count = len(completion_ids) // num_generations
        kept_prompt_ids = []
        kept_completion_ids = []
        initial_truncated_counts = []
        retry_eligible_groups = 0
        retry_rescued_groups = 0
        retry_failed_groups = 0
        retry_extra_generations = 0

        for group_idx in range(group_count):
            start = group_idx * num_generations
            end = start + num_generations
            group_prompt_ids = prompt_ids[start:end]
            group_completion_ids = completion_ids[start:end]
            truncated = [self._is_completion_truncated_ids(ids) for ids in group_completion_ids]
            truncated_count = sum(truncated)
            initial_truncated_counts.append(truncated_count)

            if truncated_count == 0 or truncated_count > self.max_retry_truncated_count:
                kept_prompt_ids.extend(group_prompt_ids)
                kept_completion_ids.extend(group_completion_ids)
                continue

            retry_eligible_groups += 1
            accepted_pairs = [
                (p_ids, c_ids)
                for p_ids, c_ids, is_truncated in zip(group_prompt_ids, group_completion_ids, truncated, strict=True)
                if not is_truncated
            ]

            retry_prompt = prompts[group_idx]
            for _ in range(self.max_truncation_retry_rounds):
                needed = num_generations - len(accepted_pairs)
                if needed <= 0:
                    break
                retry_generated = self._generate_retry_batch([retry_prompt], needed)
                retry_prompt_ids, retry_completion_ids, _, _, _, retry_logprobs, retry_extra_fields, *_ = (
                    retry_generated
                )
                if retry_logprobs is not None:
                    break
                if retry_extra_fields:
                    extra_fields.update(retry_extra_fields)
                retry_extra_generations += len(retry_completion_ids)
                for p_ids, c_ids in zip(retry_prompt_ids, retry_completion_ids, strict=True):
                    if not self._is_completion_truncated_ids(c_ids):
                        accepted_pairs.append((p_ids, c_ids))
                        if len(accepted_pairs) >= num_generations:
                            break

            if len(accepted_pairs) >= num_generations:
                retry_rescued_groups += 1
                final_pairs = accepted_pairs[:num_generations]
                kept_prompt_ids.extend(p_ids for p_ids, _ in final_pairs)
                kept_completion_ids.extend(c_ids for _, c_ids in final_pairs)
            else:
                retry_failed_groups += 1
                kept_prompt_ids.extend(group_prompt_ids)
                kept_completion_ids.extend(group_completion_ids)

        self._metrics[mode]["truncation_retry/eligible_groups"].append(float(retry_eligible_groups))
        self._metrics[mode]["truncation_retry/rescued_groups"].append(float(retry_rescued_groups))
        self._metrics[mode]["truncation_retry/failed_groups"].append(float(retry_failed_groups))
        self._metrics[mode]["truncation_retry/extra_generations"].append(float(retry_extra_generations))
        if group_count > 0:
            self._metrics[mode]["truncation_retry/initial_truncated_completion_frac"].append(
                sum(initial_truncated_counts) / (group_count * num_generations)
            )

        return kept_prompt_ids, kept_completion_ids, logprobs, extra_fields

    def _generate(self, prompts: list):
        generated = super()._generate(prompts)
        prompt_ids, completion_ids, tool_mask, completions, num_items_in_batch, logprobs, extra_fields, *extra_returns = (
            generated
        )
        if tool_mask is not None:
            return generated

        prompt_ids, completion_ids, logprobs, extra_fields = self._retry_truncated_completions(
            copy.deepcopy(prompts),
            prompt_ids,
            completion_ids,
            logprobs,
            extra_fields,
        )

        if logprobs is not None:
            return (
                prompt_ids,
                completion_ids,
                tool_mask,
                completions,
                num_items_in_batch,
                logprobs,
                extra_fields,
                *extra_returns,
            )

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        if completion_ids:
            completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        else:
            completion_lengths = torch.zeros(0, device=device, dtype=torch.long)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_completion_tokens = agg_completion_lengths.sum()

        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        if mode == "train":
            original_num_items = num_items_in_batch.item() if hasattr(num_items_in_batch, "item") else num_items_in_batch
            self.state.num_input_tokens_seen += (total_completion_tokens - original_num_items).item()
            self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        eos_and_pad = {self.pad_token_id}
        eos_token_id = getattr(self, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple, set)):
            eos_and_pad.update(eos_token_id)
        else:
            eos_and_pad.add(eos_token_id)

        is_truncated = torch.tensor(
            [bool(ids) and int(ids[-1]) not in eos_and_pad for ids in completion_ids],
            device=device,
            dtype=torch.bool,
        )
        agg_is_truncated = self.accelerator.gather(is_truncated)
        if agg_completion_lengths.numel() > 0:
            self._metrics[mode]["completions/mean_length"][-1] = agg_completion_lengths.float().mean().item()
            self._metrics[mode]["completions/min_length"][-1] = agg_completion_lengths.float().min().item()
            self._metrics[mode]["completions/max_length"][-1] = agg_completion_lengths.float().max().item()
            self._metrics[mode]["completions/clipped_ratio"][-1] = agg_is_truncated.float().mean().item()
            term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
            if len(term_completion_lengths) == 0:
                term_completion_lengths = torch.zeros(1, device=device)
            self._metrics[mode]["completions/mean_terminated_length"][-1] = (
                term_completion_lengths.float().mean().item()
            )
            self._metrics[mode]["completions/min_terminated_length"][-1] = (
                term_completion_lengths.float().min().item()
            )
            self._metrics[mode]["completions/max_terminated_length"][-1] = (
                term_completion_lengths.float().max().item()
            )

        if prompts and isinstance(prompts[0], list):
            contents = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            completions = [[{"role": "assistant", "content": content}] for content in contents]
        else:
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        return (
            prompt_ids,
            completion_ids,
            None,
            completions,
            total_completion_tokens,
            logprobs,
            extra_fields,
            *extra_returns,
        )

    def _recent_correctness_rewards(self, batch_size: int, device: torch.device):
        correctness_reward_name = self._find_correctness_reward_name()
        if correctness_reward_name is None:
            return None
        rewards = self._get_recent_reward_values(
            reward_name=correctness_reward_name,
            batch_size=batch_size,
            device=device,
        )
        if rewards is None:
            return None

        local_batch_size = batch_size // self.accelerator.num_processes
        start = self.accelerator.process_index * local_batch_size
        end = start + local_batch_size
        return rewards[start:end]

    def _find_length_reward_name(self):
        for reward_name in self.reward_func_names:
            if "length" in reward_name:
                return reward_name
        return None

    def _recent_length_rewards(self, batch_size: int, device: torch.device):
        length_reward_name = self._find_length_reward_name()
        if length_reward_name is None:
            return None
        rewards = self._get_recent_reward_values(
            reward_name=length_reward_name,
            batch_size=batch_size,
            device=device,
        )
        if rewards is None:
            return None

        local_batch_size = batch_size // self.accelerator.num_processes
        start = self.accelerator.process_index * local_batch_size
        end = start + local_batch_size
        return rewards[start:end]

    def _completion_truncation_mask(self, outputs) -> Optional[torch.Tensor]:
        completion_ids = outputs.get("completion_ids")
        completion_mask = outputs.get("completion_mask")
        if completion_ids is None or completion_mask is None:
            return None
        if completion_ids.numel() == 0 or completion_ids.size(0) == 0:
            return torch.zeros(completion_ids.size(0), device=completion_ids.device, dtype=torch.bool)

        lengths = completion_mask.long().sum(dim=1)
        has_tokens = lengths > 0
        last_indices = torch.clamp(lengths - 1, min=0).unsqueeze(1)
        last_tokens = completion_ids.gather(1, last_indices).squeeze(1)

        ended = torch.zeros_like(has_tokens, dtype=torch.bool)
        eos_token_id = getattr(self, "eos_token_id", None)
        if eos_token_id is not None:
            if isinstance(eos_token_id, (list, tuple, set)):
                for token_id in eos_token_id:
                    ended |= last_tokens == int(token_id)
            else:
                ended |= last_tokens == int(eos_token_id)

        pad_token_id = getattr(self, "pad_token_id", None)
        if pad_token_id is not None:
            ended |= last_tokens == int(pad_token_id)

        return has_tokens & (lengths >= self.args.max_completion_length) & ~ended

    def _generate_and_score_completions(self, inputs):
        outputs = super()._generate_and_score_completions(inputs)

        mode = "train" if self.model.training else "eval"
        if mode != "train":
            return outputs

        advantages = outputs["advantages"]
        num_generations = self.num_generations
        if num_generations <= 0 or advantages.numel() % num_generations != 0:
            return outputs

        correctness_rewards = self._recent_correctness_rewards(
            batch_size=advantages.numel() * self.accelerator.num_processes,
            device=advantages.device,
        )
        if correctness_rewards is None or correctness_rewards.numel() != advantages.numel():
            return outputs

        grouped_correctness = correctness_rewards.view(-1, num_generations)
        correct_rates = grouped_correctness.mean(dim=1)
        accepted_groups = (correct_rates >= self.min_correct_rate) & (correct_rates <= self.max_correct_rate)

        truncation_mask = self._completion_truncation_mask(outputs)
        truncated_groups = None
        if truncation_mask is not None and truncation_mask.numel() == advantages.numel():
            grouped_truncation = truncation_mask.view(-1, num_generations)
            truncated_groups = grouped_truncation.any(dim=1)
            accepted_groups = accepted_groups & ~truncated_groups

        length_ok_groups = torch.ones_like(accepted_groups)
        too_short_groups = torch.zeros_like(accepted_groups)
        too_long_groups = torch.zeros_like(accepted_groups)
        completion_mask_for_lengths = outputs.get("completion_mask")
        if completion_mask_for_lengths is not None:
            completion_lengths = completion_mask_for_lengths.long().sum(dim=1).view(-1, num_generations)
            too_short_groups = (completion_lengths < self.accepted_length_min).any(dim=1)
            if self.accepted_length_max is not None:
                too_long_groups = (completion_lengths > self.accepted_length_max).any(dim=1)
            length_ok_groups = ~(too_short_groups | too_long_groups)
            accepted_groups = accepted_groups & length_ok_groups

        length_only_groups = torch.zeros_like(accepted_groups)
        length_only_advantages = None
        length_only_sample_mask = torch.zeros_like(advantages, dtype=torch.bool)
        length_penalized_groups = torch.zeros_like(accepted_groups)
        if self.rejected_length_only_update:
            length_rewards = self._recent_length_rewards(
                batch_size=advantages.numel() * self.accelerator.num_processes,
                device=advantages.device,
            )
            if length_rewards is not None and length_rewards.numel() == advantages.numel():
                grouped_length_rewards = length_rewards.view(-1, num_generations)
                length_penalized_groups = (grouped_length_rewards < 0).any(dim=1)

                length_only_groups = (~accepted_groups) & length_penalized_groups
                grouped_length_only_sample_mask = length_only_groups.unsqueeze(1) & (grouped_length_rewards < 0)
                length_only_sample_mask = grouped_length_only_sample_mask.reshape_as(advantages)
                length_only_advantages = (
                    grouped_length_rewards.clamp(max=0.0).reshape_as(advantages)
                    * self.rejected_length_only_advantage_scale
                )

        accuracy_gate_mask = accepted_groups.repeat_interleave(num_generations).to(advantages.device)
        loss_gate_mask = (accuracy_gate_mask.bool() | length_only_sample_mask.bool()).float()

        outputs["accuracy_gate_mask"] = accuracy_gate_mask.float()
        outputs["loss_gate_mask"] = loss_gate_mask
        if length_only_advantages is not None:
            outputs["advantages"] = (
                advantages * accuracy_gate_mask.float()
                + length_only_advantages * length_only_sample_mask.float()
            )
        else:
            outputs["advantages"] = advantages * accuracy_gate_mask.float()

        accepted_count = int(accepted_groups.sum().item())
        length_only_count = int(length_only_groups.sum().item())
        group_count = int(accepted_groups.numel())
        self.accepted_update_steps += accepted_count
        self.rejected_update_steps += group_count - accepted_count
        attempted_samples = self.accepted_update_steps + self.rejected_update_steps
        acceptance_rate = self.accepted_update_steps / max(attempted_samples, 1)
        self._metrics[mode]["accuracy_gate/accepted_group_frac"].append(accepted_count / max(group_count, 1))
        self._metrics[mode]["accuracy_gate/accepted_groups"].append(float(accepted_count))
        self._metrics[mode]["accuracy_gate/group_count"].append(float(group_count))
        self._metrics[mode]["accuracy_gate/correct_rate_mean"].append(correct_rates.mean().item())
        self._metrics[mode]["accuracy_gate/correct_rate_min"].append(correct_rates.min().item())
        self._metrics[mode]["accuracy_gate/correct_rate_max"].append(correct_rates.max().item())
        self._metrics[mode]["accuracy_gate/accepted_update_steps"].append(float(self.accepted_update_steps))
        self._metrics[mode]["accuracy_gate/rejected_update_steps"].append(float(self.rejected_update_steps))
        self._metrics[mode]["effective_step"].append(float(self.accepted_update_steps))
        self._metrics[mode]["attempted_samples"].append(float(attempted_samples))
        self._metrics[mode]["accepted_samples"].append(float(self.accepted_update_steps))
        self._metrics[mode]["acceptance_rate"].append(acceptance_rate)
        self._metrics[mode]["accepted_length_gate/group_frac"].append(length_ok_groups.float().mean().item())
        self._metrics[mode]["accepted_length_gate/too_short_group_frac"].append(
            too_short_groups.float().mean().item()
        )
        self._metrics[mode]["accepted_length_gate/too_long_group_frac"].append(
            too_long_groups.float().mean().item()
        )
        self._metrics[mode]["length_only_update/groups"].append(float(length_only_count))
        self._metrics[mode]["length_only_update/group_frac"].append(length_only_count / max(group_count, 1))
        self._metrics[mode]["length_only_update/penalized_group_frac"].append(
            length_penalized_groups.float().mean().item()
        )
        self._metrics[mode]["length_only_update/penalized_completion_frac"].append(
            length_only_sample_mask.float().mean().item()
        )
        if truncated_groups is not None:
            truncated_count = int(truncated_groups.sum().item())
            self._metrics[mode]["truncation/truncated_group_frac"].append(truncated_count / max(group_count, 1))
            self._metrics[mode]["truncation/truncated_groups"].append(float(truncated_count))
            self._metrics[mode]["truncation/truncated_completion_frac"].append(
                truncation_mask.float().mean().item()
            )
        return outputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        gate_mask = inputs.get("accuracy_gate_mask")
        self._last_train_step_has_accepted_group = bool(
            self.model.training and gate_mask is not None and gate_mask.detach().bool().any().item()
        )
        loss_gate_mask = inputs.get("loss_gate_mask", gate_mask)
        if self.model.training and loss_gate_mask is not None and "completion_mask" in inputs:
            inputs = dict(inputs)
            token_gate_mask = loss_gate_mask.to(
                device=inputs["completion_mask"].device,
                dtype=inputs["completion_mask"].dtype,
            )
            while token_gate_mask.dim() < inputs["completion_mask"].dim():
                token_gate_mask = token_gate_mask.unsqueeze(-1)
            inputs["completion_mask"] = inputs["completion_mask"] * token_gate_mask
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def training_step(self, model, inputs, num_items_in_batch):
        output = super().training_step(model, inputs, num_items_in_batch)
        return output


class ScheduledAccuracyGateGRPOTrainer(AccuracyGateGRPOTrainer):
    def __init__(self, *args, accuracy_gate_stages: list[AccuracyGateStage], **kwargs):
        if not accuracy_gate_stages:
            raise ValueError("accuracy_gate_stages must not be empty.")
        first_stage = accuracy_gate_stages[0]
        kwargs["min_correct_rate"] = first_stage.min_correct_rate
        kwargs["max_correct_rate"] = first_stage.max_correct_rate
        super().__init__(*args, **kwargs)
        self.accuracy_gate_stages = accuracy_gate_stages
        self._last_logged_stage_index = None

    def current_stage_index(self) -> int:
        accepted_steps = self.accepted_update_steps
        for idx, stage in enumerate(self.accuracy_gate_stages):
            if accepted_steps < stage.end_step:
                return idx
        return len(self.accuracy_gate_stages) - 1

    def current_stage(self) -> AccuracyGateStage:
        return self.accuracy_gate_stages[self.current_stage_index()]

    def _sync_current_stage(self, mode: str):
        stage_index = self.current_stage_index()
        stage = self.accuracy_gate_stages[stage_index]
        self.min_correct_rate = stage.min_correct_rate
        self.max_correct_rate = stage.max_correct_rate

        if self._last_logged_stage_index != stage_index:
            print(
                "\n切换 accuracy gate 阶段: "
                f"stage={stage_index + 1}/{len(self.accuracy_gate_stages)}, "
                f"accepted_update_steps=[{stage.start_step}, {stage.end_step}), "
                f"correct_rate=[{stage.min_correct_rate}, {stage.max_correct_rate}]"
            )
            self._last_logged_stage_index = stage_index

        self._metrics[mode]["accuracy_gate/stage_index"].append(float(stage_index + 1))
        self._metrics[mode]["accuracy_gate/stage_start_step"].append(float(stage.start_step))
        self._metrics[mode]["accuracy_gate/stage_end_step"].append(float(stage.end_step))
        self._metrics[mode]["accuracy_gate/stage_min_correct_rate"].append(stage.min_correct_rate)
        self._metrics[mode]["accuracy_gate/stage_max_correct_rate"].append(stage.max_correct_rate)

    def _generate_and_score_completions(self, inputs):
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            self._sync_current_stage(mode)
        return super()._generate_and_score_completions(inputs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./grpo_qwen25_15b_gsm8k_lora_accuracy_gate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--train_scores_file", type=str, default=None)
    parser.add_argument(
        "--shuffle_train_scores",
        action="store_true",
        help="Shuffle rows from --train_scores_file with --seed before building the training dataset.",
    )
    parser.add_argument(
        "--train_scores_shuffle_seed",
        type=int,
        default=None,
        help="Seed used only for shuffling --train_scores_file rows. Defaults to --seed.",
    )
    parser.add_argument("--min_uid1", type=int, default=None)
    parser.add_argument("--max_uid1", type=int, default=None)
    parser.add_argument("--prompt_style", type=str, default="short", choices=["short", "fewshot"])
    parser.add_argument(
        "--stage_schedule",
        type=str,
        required=True,
        help="Comma-separated stages in RANGE:STEPS format, e.g. 0.875-0.625:50,0.75-0.5:50.",
    )
    parser.add_argument("--target_update_steps", type=int, default=None)
    parser.add_argument("--max_attempt_steps", type=int, default=None)
    parser.add_argument("--min_correct_rate", type=float, default=0.375)
    parser.add_argument("--max_correct_rate", type=float, default=0.625)
    parser.add_argument("--accepted_length_min", type=int, default=0)
    parser.add_argument("--accepted_length_max", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--max_completion_length", type=int, default=448)
    parser.add_argument("--length_reward_weight", type=float, default=0.0)
    parser.add_argument("--length_reward_good_min", type=int, default=None)
    parser.add_argument("--length_reward_good_max", type=int, default=None)
    parser.add_argument("--length_reward_short_min", type=int, default=0)
    parser.add_argument("--length_reward_max_penalty", type=float, default=1.0)
    parser.add_argument("--rejected_length_only_update", type=int, default=0, choices=[0, 1])
    parser.add_argument("--rejected_length_only_advantage_scale", type=float, default=None)
    parser.add_argument("--max_retry_truncated_count", type=int, default=6)
    parser.add_argument("--max_truncation_retry_rounds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=5)
    parser.add_argument("--save_total_limit", type=int, default=1000)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--init_adapter_path", type=str, default=None)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    accuracy_gate_stages = parse_stage_schedule(args.stage_schedule)
    schedule_update_steps = accuracy_gate_stages[-1].end_step
    if args.target_update_steps is None:
        args.target_update_steps = schedule_update_steps
    if args.target_update_steps <= 0:
        raise ValueError("--target_update_steps must be positive.")
    if args.target_update_steps > schedule_update_steps:
        raise ValueError(
            "--target_update_steps cannot exceed total scheduled update steps. "
            f"Got target_update_steps={args.target_update_steps}, schedule_total={schedule_update_steps}."
        )
    if args.accepted_length_min < 0:
        raise ValueError("--accepted_length_min must be non-negative.")
    if args.accepted_length_max is not None and args.accepted_length_max < args.accepted_length_min:
        raise ValueError("--accepted_length_max must be >= --accepted_length_min.")
    if args.accepted_length_max is not None and args.accepted_length_max > args.max_completion_length:
        raise ValueError("--accepted_length_max must be <= --max_completion_length.")
    if args.length_reward_weight < 0:
        raise ValueError("--length_reward_weight must be non-negative.")
    if args.length_reward_max_penalty < 0:
        raise ValueError("--length_reward_max_penalty must be non-negative.")
    if args.rejected_length_only_update and args.length_reward_weight <= 0:
        print("Warning: length reward is disabled, so rejected length-only update is also disabled.")
        args.rejected_length_only_update = 0
    if args.length_reward_short_min < 0:
        raise ValueError("--length_reward_short_min must be non-negative.")
    if args.rejected_length_only_advantage_scale is None:
        args.rejected_length_only_advantage_scale = args.length_reward_weight
    if args.rejected_length_only_advantage_scale < 0:
        raise ValueError("--rejected_length_only_advantage_scale must be non-negative.")

    length_reward_good_min = args.length_reward_good_min if args.length_reward_good_min is not None else 140
    length_reward_good_max = (
        args.length_reward_good_max
        if args.length_reward_good_max is not None
        else 280
    )
    if length_reward_good_min < args.length_reward_short_min:
        raise ValueError("--length_reward_good_min must be >= --length_reward_short_min.")
    if length_reward_good_max < length_reward_good_min:
        raise ValueError("--length_reward_good_max must be >= --length_reward_good_min.")
    if length_reward_good_max >= args.max_completion_length:
        raise ValueError("--length_reward_good_max must be smaller than --max_completion_length.")

    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if effective_batch_size % args.num_generations != 0:
        raise ValueError(
            "effective batch size must be divisible by num_generations. "
            f"Got per_device_train_batch_size={args.per_device_train_batch_size}, "
            f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
            f"num_generations={args.num_generations}."
        )
    if args.beta != 0.0:
        print(
            "KL regularization enabled; accepted groups and rejected length-only groups enter KL, "
            "other rejected groups are masked out."
        )

    initial_accepted_update_steps = checkpoint_step(args.resume_from_checkpoint)
    resume_attempt_step = checkpoint_trainer_global_step(args.resume_from_checkpoint)
    max_attempt_steps = args.max_attempt_steps or args.target_update_steps * 20
    trainer_max_steps = resume_attempt_step + max_attempt_steps

    print("模型初始化，加载数据中...")
    print(
        "Scheduled accuracy gate: "
        f"target_update_steps={args.target_update_steps}, max_attempt_steps={max_attempt_steps}"
    )
    print("Accuracy gate stages:")
    for idx, stage in enumerate(accuracy_gate_stages, start=1):
        print(
            f"  stage {idx}: accepted_update_steps=[{stage.start_step}, {stage.end_step}), "
            f"correct_rate=[{stage.min_correct_rate}, {stage.max_correct_rate}], "
            f"stage_update_steps={stage.update_steps}"
        )
    print(
        "Accepted length gate: "
        f"min_length={args.accepted_length_min}, "
        f"max_length={args.accepted_length_max if args.accepted_length_max is not None else 'disabled'}"
    )
    print("Truncation gate: reject any prompt group with a length-truncated completion")
    print(
        "Truncation retry: "
        f"max_retry_truncated_count={args.max_retry_truncated_count}, "
        f"max_truncation_retry_rounds={args.max_truncation_retry_rounds}"
    )
    if args.length_reward_weight > 0:
        print(
            "Length reward: "
            f"weight={args.length_reward_weight}, good_range=[{length_reward_good_min}, {length_reward_good_max}], "
            f"short_min={args.length_reward_short_min}, "
            f"max_penalty={args.length_reward_max_penalty}, max_completion_length={args.max_completion_length}"
        )
    else:
        print("Length reward: disabled")
    print(
        "Rejected length-only update: "
        f"{'enabled' if args.rejected_length_only_update else 'disabled'}, "
        f"advantage_scale={args.rejected_length_only_advantage_scale}"
    )
    if args.resume_from_checkpoint:
        print(
            f"从 checkpoint 续训: {args.resume_from_checkpoint}; "
            f"initial_accepted_update_steps={initial_accepted_update_steps}; "
            f"resume_attempt_step={resume_attempt_step}; "
            f"trainer max_steps={trainer_max_steps}"
        )

    train_dataset = build_grpo_dataset(
        split=args.train_split,
        dataset_path=args.dataset_path,
        max_samples=args.train_samples if args.train_samples > 0 else None,
        seed=args.train_scores_shuffle_seed if args.train_scores_shuffle_seed is not None else args.seed,
        selected_rows_path=args.train_scores_file,
        min_uid1=args.min_uid1,
        max_uid1=args.max_uid1,
        prompt_style=args.prompt_style,
        shuffle_selected_rows=args.shuffle_train_scores,
    )

    tokenizer = load_tokenizer(args.model_name)
    print("初始化模型中...")
    model = load_model_for_training(model_name=args.model_name, use_4bit=args.use_4bit)

    if args.init_adapter_path is not None:
        print(f"加载初始 LoRA adapter: {args.init_adapter_path}")
        model = PeftModel.from_pretrained(model, args.init_adapter_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        )
        model = get_peft_model(model, lora_config)

    resolved_base_model_path = resolve_cached_model_path(args.model_name)
    for peft_config in model.peft_config.values():
        peft_config.base_model_name_or_path = resolved_base_model_path
    model.print_trainable_parameters()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    reward_funcs = [gsm8k_correctness_reward, final_answer_format_reward]
    reward_weights = [1.0, 0.1]
    if args.length_reward_weight > 0:
        reward_funcs.append(
            CompletionTokenLengthReward(
                max_length=args.max_completion_length,
                good_min=length_reward_good_min,
                good_max=length_reward_good_max,
                short_min=args.length_reward_short_min,
                max_penalty=args.length_reward_max_penalty,
            )
        )
        reward_weights.append(args.length_reward_weight)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=trainer_max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy="no",
        save_total_limit=args.save_total_limit,
        disable_tqdm=True,
        report_to=args.report_to,
        seed=args.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.0,
        beta=args.beta,
        epsilon=args.epsilon,
        num_iterations=1,
        scale_rewards="group",
        loss_type="dapo",
        reward_weights=reward_weights,
        log_completions=False,
    )

    trainer = ScheduledAccuracyGateGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        accuracy_gate_stages=accuracy_gate_stages,
        accepted_length_min=args.accepted_length_min,
        accepted_length_max=args.accepted_length_max,
        rejected_length_only_update=bool(args.rejected_length_only_update),
        rejected_length_only_advantage_scale=args.rejected_length_only_advantage_scale,
        max_retry_truncated_count=args.max_retry_truncated_count,
        max_truncation_retry_rounds=args.max_truncation_retry_rounds,
    )
    trainer.accepted_update_steps = initial_accepted_update_steps
    trainer.add_callback(StopAfterAcceptedUpdatesCallback(lambda: trainer, args.target_update_steps))
    trainer.add_callback(
        AcceptedUpdateProgressAndCheckpointCallback(
            lambda: trainer,
            target_update_steps=args.target_update_steps,
            save_steps=args.save_steps,
            initial_accepted_update_steps=initial_accepted_update_steps,
        )
    )

    print("开始训练...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"模型已保存到 {args.output_dir}")
    print(
        "Accuracy gate summary: "
        f"accepted_update_steps={trainer.accepted_update_steps}, "
        f"rejected_update_steps={trainer.rejected_update_steps}, "
        f"target_update_steps={args.target_update_steps}"
    )


if __name__ == "__main__":
    main()
