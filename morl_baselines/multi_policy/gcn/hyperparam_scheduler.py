import math


class HyperparamScheduler:
    """Temporal curriculum scheduler for lcn_lambda,
    extended to handle any applicable float hyperparam.

    Controls the transition from Pareto exploration (start_val) to
    Lorenz fairness (end_val) over training.
    """

    def __init__(
        self,
        schedule_type: str,
        target_key: str,
        start_val: float,
        end_val: float,
        total_timesteps: int,
        warmup_fraction: float = 0.0,
        freeze_fraction: float = 0.1,
    ):
        """
        Args:
            schedule_type: one of 'linear', 'cosine', 'step'.
            target_key: the key of the model's hyperparameters to perform scheduling on
            start_val: initial hyperparameter value (lambda: typically 1.0 for Pareto exploration).
            end_val: final hyperparameter value (e.g. lambda: 0.0 for full Lorenz fairness).
            total_timesteps: total training timesteps.
            warmup_fraction: fraction of training to keep the hyperparameter at start_val.
            freeze_fraction: fraction of training at end to keep the hyperparameter at end_val.
        """
        assert schedule_type in ('linear', 'cosine', 'step'), \
            f"Unknown schedule_type: {schedule_type}. Use 'linear', 'cosine', or 'step'."
        assert 0 <= warmup_fraction < 1, "warmup_fraction must be in [0, 1)"
        assert 0 <= freeze_fraction < 1, "freeze_fraction must be in [0, 1)"
        assert warmup_fraction + freeze_fraction < 1, "Cannot warmup and freeze for longer than the entire training process. Please ensure warmup + freeze < 1"

        self.schedule_type = schedule_type
        self.start_val = start_val
        self.end_val = end_val
        self.target_key = target_key
        self.total_timesteps = total_timesteps
        self.warmup_fraction = warmup_fraction
        self.freeze_fraction = freeze_fraction

        self._warmup_end = int(total_timesteps * warmup_fraction)
        self._freeze_start = int(total_timesteps * (1 - freeze_fraction))
        self._active_duration = max(self._freeze_start - self._warmup_end, 1)

    def step(self, current_step: int, params) -> float:
        """Return the base lambda value at the given training step."""
        if current_step <= self._warmup_end:
            params[self.target_key] = self.start_val
        if current_step >= self._freeze_start:
            params[self.target_key] = self.end_val

        progress = (current_step - self._warmup_end) / self._active_duration
        progress = max(0.0, min(1.0, progress))

        if self.schedule_type == 'linear':
            params[self.target_key] = self.start_val + (self.end_val - self.start_val) * progress
            return
        elif self.schedule_type == 'cosine':
            params[self.target_key] = self.end_val + 0.5 * (self.start_val - self.end_val) * (1 + math.cos(math.pi * progress))
        elif self.schedule_type == 'step':
            if progress < 0.25:
                params[self.target_key] = self.start_val
            elif progress < 0.5:
                params[self.target_key] = self.start_val + (self.end_val - self.start_val) * 0.33
            elif progress < 0.75:
                params[self.target_key] = self.start_val + (self.end_val - self.start_val) * 0.67
            else:
                params[self.target_key] = self.end_val

    def get_config(self) -> dict:
        return {
            "scheduling_type": self.schedule_type,
            "start_val": self.start_val,
            "end_val": self.end_val,
            "warmup_fraction": self.warmup_fraction,
            "freeze_fraction": self.freeze_fraction,
        }
