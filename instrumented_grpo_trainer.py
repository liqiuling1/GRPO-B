import torch
from trl import GRPOTrainer


class InstrumentedGRPOTrainer(GRPOTrainer):
    """Add advantage diagnostics to the standard GRPO training logs."""

    def _get_recent_reward_values(self, reward_name, batch_size, device):
        reward_log = self._logs["rewards"].get(reward_name)
        if reward_log is None or len(reward_log) < batch_size:
            return None
        recent_values = list(reward_log)[-batch_size:]
        return torch.tensor(recent_values, device=device, dtype=torch.float32)

    def _find_correctness_reward_name(self):
        for reward_name in self.reward_func_names:
            if "correctness" in reward_name:
                return reward_name
        return None

    def _generate_and_score_completions(self, inputs):
        outputs = super()._generate_and_score_completions(inputs)

        mode = "train" if self.model.training else "eval"
        advantages = outputs["advantages"].detach().float()
        gathered_advantages = self.accelerator.gather(advantages)
        zero_advantages = torch.isclose(
            gathered_advantages,
            torch.zeros_like(gathered_advantages),
        )
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        self._metrics[mode]["advantage/std"].append(gathered_advantages.std().item())
        self._metrics[mode]["advantage/abs_mean"].append(gathered_advantages.abs().mean().item())
        self._metrics[mode]["advantage/zero_frac"].append(zero_advantages.float().mean().item())

        if num_generations > 0 and gathered_advantages.numel() % num_generations == 0:
            grouped_advantages = gathered_advantages.view(-1, num_generations)
            zero_advantage_groups = torch.isclose(
                grouped_advantages.abs().amax(dim=1),
                torch.zeros(grouped_advantages.size(0), device=grouped_advantages.device),
            )
            self._metrics[mode]["prompt/advantage_zero_group_frac"].append(zero_advantage_groups.float().mean().item())

            correctness_reward_name = self._find_correctness_reward_name()
            correctness_rewards = self._get_recent_reward_values(
                reward_name=correctness_reward_name,
                batch_size=gathered_advantages.numel(),
                device=gathered_advantages.device,
            )
            if correctness_rewards is not None:
                grouped_correctness = correctness_rewards.view(-1, num_generations)
                correctness_group_std = grouped_correctness.std(dim=1)
                zero_std_groups = torch.isclose(
                    correctness_group_std,
                    torch.zeros_like(correctness_group_std),
                )
                centered_correctness = grouped_correctness - grouped_correctness.mean(dim=1, keepdim=True)
                normalized_correctness_advantages = centered_correctness / (
                    correctness_group_std.unsqueeze(1) + 1e-4
                )
                zero_correctness_advantage_groups = torch.isclose(
                    normalized_correctness_advantages.abs().amax(dim=1),
                    torch.zeros(grouped_correctness.size(0), device=grouped_correctness.device),
                )

                self._metrics[mode]["prompt/correctness_reward_std_mean"].append(correctness_group_std.mean().item())
                self._metrics[mode]["prompt/correctness_reward_zero_std_frac"].append(
                    zero_std_groups.float().mean().item()
                )
                self._metrics[mode]["prompt/correctness_adv_zero_group_frac"].append(
                    zero_correctness_advantage_groups.float().mean().item()
                )

        return outputs

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        logs = dict(logs)
        grad_norm = logs.get("grad_norm")
        learning_rate = logs.get("learning_rate")
        if grad_norm is not None and learning_rate is not None:
            logs["update_proxy"] = float(grad_norm) * float(learning_rate)
        super().log(logs, start_time)
