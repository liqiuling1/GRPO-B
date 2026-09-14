import argparse
import os
import random
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from trl import GRPOConfig

from dataset_utils import build_grpo_dataset
from instrumented_grpo_trainer import InstrumentedGRPOTrainer
from model_utils import load_model_for_training, load_tokenizer, resolve_cached_model_path
from reward_utils import CompletionTokenLengthReward, final_answer_format_reward, gsm8k_correctness_reward


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SafeOrderedGRPOTrainer(InstrumentedGRPOTrainer):
    """Original GRPO with group-level length/truncation safety masks.

    Safe groups use the normal GRPO advantages. Unsafe groups do not use
    correctness/format advantages; optionally, only completions with negative
    length reward get a length-only penalty update.
    """

    def __init__(
        self,
        *args,
        safe_length_min: int = 0,
        safe_length_max: Optional[int] = None,
        rejected_length_only_update: bool = True,
        rejected_length_only_advantage_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.safe_length_min = safe_length_min
        self.safe_length_max = safe_length_max
        self.rejected_length_only_update = rejected_length_only_update
        self.rejected_length_only_advantage_scale = rejected_length_only_advantage_scale

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

        group_count = advantages.numel() // num_generations
        device = advantages.device
        safe_groups = torch.ones(group_count, device=device, dtype=torch.bool)
        too_short_groups = torch.zeros(group_count, device=device, dtype=torch.bool)
        too_long_groups = torch.zeros(group_count, device=device, dtype=torch.bool)
        truncated_groups = torch.zeros(group_count, device=device, dtype=torch.bool)

        completion_mask = outputs.get("completion_mask")
        if completion_mask is not None:
            completion_lengths = completion_mask.long().sum(dim=1).view(-1, num_generations)
            if self.safe_length_min > 0:
                too_short_groups = (completion_lengths < self.safe_length_min).any(dim=1)
            if self.safe_length_max is not None:
                too_long_groups = (completion_lengths > self.safe_length_max).any(dim=1)
            safe_groups = safe_groups & ~too_short_groups & ~too_long_groups

        truncation_mask = self._completion_truncation_mask(outputs)
        if truncation_mask is not None and truncation_mask.numel() == advantages.numel():
            truncated_groups = truncation_mask.view(-1, num_generations).any(dim=1)
            safe_groups = safe_groups & ~truncated_groups

        length_only_sample_mask = torch.zeros_like(advantages, dtype=torch.bool)
        length_only_advantages = None
        length_penalized_groups = torch.zeros(group_count, device=device, dtype=torch.bool)
        if self.rejected_length_only_update:
            length_rewards = self._recent_length_rewards(
                batch_size=advantages.numel() * self.accelerator.num_processes,
                device=device,
            )
            if length_rewards is not None and length_rewards.numel() == advantages.numel():
                grouped_length_rewards = length_rewards.view(-1, num_generations)
                unsafe_groups = ~safe_groups
                length_penalized_groups = (grouped_length_rewards < 0).any(dim=1)
                grouped_length_only_sample_mask = unsafe_groups.unsqueeze(1) & (grouped_length_rewards < 0)
                length_only_sample_mask = grouped_length_only_sample_mask.reshape_as(advantages)
                length_only_advantages = (
                    grouped_length_rewards.clamp(max=0.0).reshape_as(advantages)
                    * self.rejected_length_only_advantage_scale
                )

        normal_sample_mask = safe_groups.repeat_interleave(num_generations)
        loss_gate_mask = (normal_sample_mask | length_only_sample_mask).float()
        outputs["safe_ordered_normal_mask"] = normal_sample_mask.float()
        outputs["loss_gate_mask"] = loss_gate_mask
        if length_only_advantages is not None:
            outputs["advantages"] = (
                advantages * normal_sample_mask.float()
                + length_only_advantages * length_only_sample_mask.float()
            )
        else:
            outputs["advantages"] = advantages * normal_sample_mask.float()

        safe_count = int(safe_groups.sum().item())
        unsafe_count = group_count - safe_count
        length_only_count = int(((~safe_groups) & length_penalized_groups).sum().item())
        self._metrics[mode]["safe_ordered/safe_group_frac"].append(safe_count / max(group_count, 1))
        self._metrics[mode]["safe_ordered/safe_groups"].append(float(safe_count))
        self._metrics[mode]["safe_ordered/unsafe_groups"].append(float(unsafe_count))
        self._metrics[mode]["safe_ordered/too_short_group_frac"].append(
            too_short_groups.float().mean().item()
        )
        self._metrics[mode]["safe_ordered/too_long_group_frac"].append(
            too_long_groups.float().mean().item()
        )
        self._metrics[mode]["safe_ordered/truncated_group_frac"].append(
            truncated_groups.float().mean().item()
        )
        self._metrics[mode]["safe_ordered/length_only_groups"].append(float(length_only_count))
        self._metrics[mode]["safe_ordered/length_only_group_frac"].append(
            length_only_count / max(group_count, 1)
        )
        self._metrics[mode]["safe_ordered/length_only_completion_frac"].append(
            length_only_sample_mask.float().mean().item()
        )

        return outputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_gate_mask = inputs.get("loss_gate_mask")
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./grpo_qwen25_15b_gsm8k_lora_grpo_ordered_safe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--train_scores_file", type=str, default=None)
    parser.add_argument("--min_uid1", type=int, default=None)
    parser.add_argument("--max_uid1", type=int, default=None)
    parser.add_argument("--prompt_style", type=str, default="short", choices=["short", "fewshot"])
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=448)
    parser.add_argument("--length_reward_weight", type=float, default=0.0)
    parser.add_argument("--length_reward_good_min", type=int, default=None)
    parser.add_argument("--length_reward_good_max", type=int, default=None)
    parser.add_argument("--length_reward_short_min", type=int, default=0)
    parser.add_argument("--length_reward_max_penalty", type=float, default=1.0)
    parser.add_argument("--safe_length_min", type=int, default=0)
    parser.add_argument("--safe_length_max", type=int, default=650)
    parser.add_argument("--rejected_length_only_update", type=int, default=1, choices=[0, 1])
    parser.add_argument("--rejected_length_only_advantage_scale", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
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

    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if effective_batch_size % args.num_generations != 0:
        raise ValueError(
            "effective batch size must be divisible by num_generations. "
            f"Got per_device_train_batch_size={args.per_device_train_batch_size}, "
            f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
            f"num_generations={args.num_generations}."
        )
    if args.length_reward_weight < 0:
        raise ValueError("--length_reward_weight must be non-negative.")
    if args.length_reward_short_min < 0:
        raise ValueError("--length_reward_short_min must be non-negative.")
    if args.length_reward_max_penalty < 0:
        raise ValueError("--length_reward_max_penalty must be non-negative.")
    if args.safe_length_min < 0:
        raise ValueError("--safe_length_min must be non-negative.")
    if args.safe_length_max is not None and args.safe_length_max < args.safe_length_min:
        raise ValueError("--safe_length_max must be >= --safe_length_min.")
    if args.safe_length_max is not None and args.safe_length_max > args.max_completion_length:
        raise ValueError("--safe_length_max must be <= --max_completion_length.")
    if args.rejected_length_only_update and args.length_reward_weight <= 0:
        print("Warning: length reward is disabled, so rejected length-only update is also disabled.")
        args.rejected_length_only_update = 0

    length_reward_good_min = args.length_reward_good_min if args.length_reward_good_min is not None else 60
    length_reward_good_max = args.length_reward_good_max if args.length_reward_good_max is not None else 650
    if length_reward_good_min < args.length_reward_short_min:
        raise ValueError("--length_reward_good_min must be >= --length_reward_short_min.")
    if length_reward_good_max < length_reward_good_min:
        raise ValueError("--length_reward_good_max must be >= --length_reward_good_min.")
    if length_reward_good_max >= args.max_completion_length:
        raise ValueError("--length_reward_good_max must be smaller than --max_completion_length.")
    if args.rejected_length_only_advantage_scale is None:
        args.rejected_length_only_advantage_scale = args.length_reward_weight
    if args.rejected_length_only_advantage_scale < 0:
        raise ValueError("--rejected_length_only_advantage_scale must be non-negative.")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    print("模型初始化，加载数据中...")
    print(
        "Safe ordered GRPO: "
        f"safe_length_min={args.safe_length_min}, "
        f"safe_length_max={args.safe_length_max}, "
        "truncate_group_normal_update=enabled, "
        f"rejected_length_only_update={bool(args.rejected_length_only_update)}, "
        f"length_only_advantage_scale={args.rejected_length_only_advantage_scale}"
    )
    if args.length_reward_weight > 0:
        print(
            "Length reward: "
            f"weight={args.length_reward_weight}, "
            f"good_range=[{length_reward_good_min}, {length_reward_good_max}], "
            f"short_min={args.length_reward_short_min}, "
            f"max_penalty={args.length_reward_max_penalty}, "
            f"max_completion_length={args.max_completion_length}"
        )
    else:
        print("Length reward: disabled")

    train_dataset = build_grpo_dataset(
        split=args.train_split,
        dataset_path=args.dataset_path,
        max_samples=args.train_samples if args.train_samples > 0 else None,
        seed=args.seed,
        selected_rows_path=args.train_scores_file,
        min_uid1=args.min_uid1,
        max_uid1=args.max_uid1,
        prompt_style=args.prompt_style,
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
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_drop_last=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
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

    trainer = SafeOrderedGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        safe_length_min=args.safe_length_min,
        safe_length_max=args.safe_length_max,
        rejected_length_only_update=bool(args.rejected_length_only_update),
        rejected_length_only_advantage_scale=args.rejected_length_only_advantage_scale,
    )

    print("开始训练...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"模型已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
