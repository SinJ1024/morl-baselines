import math


class LambdaScheduler:
    """Temporal curriculum scheduler for lcn_lambda.

    Controls the transition from Pareto exploration (lambda_start) to
    Lorenz fairness (lambda_end) over training.
    """

    def __init__(
        self,
        schedule_type: str,
        lambda_start: float,
        lambda_end: float,
        total_timesteps: int,
        warmup_fraction: float = 0.0,
        freeze_fraction: float = 0.1,
    ):
        """
        Args:
            schedule_type: one of 'linear', 'cosine', 'step'.
            lambda_start: initial lambda value (typically 1.0 for Pareto exploration).
            lambda_end: target lambda value (e.g. 0.0 for full Lorenz fairness).
            total_timesteps: total training timesteps.
            warmup_fraction: fraction of training to keep lambda at lambda_start.
            freeze_fraction: fraction of training at end to keep lambda at lambda_end.
        """
        assert schedule_type in ('linear', 'cosine', 'step'), \
            f"Unknown schedule_type: {schedule_type}. Use 'linear', 'cosine', or 'step'."
        assert 0 <= warmup_fraction < 1, "warmup_fraction must be in [0, 1)"
        assert 0 <= freeze_fraction < 1, "freeze_fraction must be in [0, 1)"
        assert warmup_fraction + freeze_fraction < 1, "warmup + freeze must be < 1"

        self.schedule_type = schedule_type
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.total_timesteps = total_timesteps
        self.warmup_fraction = warmup_fraction
        self.freeze_fraction = freeze_fraction

        self._warmup_end = int(total_timesteps * warmup_fraction)
        self._freeze_start = int(total_timesteps * (1 - freeze_fraction))
        self._active_duration = max(self._freeze_start - self._warmup_end, 1)

    def get_base_lambda(self, current_step: int) -> float:
        """Return the base lambda value at the given training step."""
        if current_step <= self._warmup_end:
            return self.lambda_start
        if current_step >= self._freeze_start:
            return self.lambda_end

        progress = (current_step - self._warmup_end) / self._active_duration
        progress = max(0.0, min(1.0, progress))

        if self.schedule_type == 'linear':
            return self.lambda_start + (self.lambda_end - self.lambda_start) * progress
        elif self.schedule_type == 'cosine':
            return self.lambda_end + 0.5 * (self.lambda_start - self.lambda_end) * (1 + math.cos(math.pi * progress))
        elif self.schedule_type == 'step':
            if progress < 0.25:
                return self.lambda_start
            elif progress < 0.5:
                return self.lambda_start + (self.lambda_end - self.lambda_start) * 0.33
            elif progress < 0.75:
                return self.lambda_start + (self.lambda_end - self.lambda_start) * 0.67
            else:
                return self.lambda_end

    def get_config(self) -> dict:
        return {
            "lambda_schedule": self.schedule_type,
            "lambda_start": self.lambda_start,
            "lambda_end": self.lambda_end,
            "warmup_fraction": self.warmup_fraction,
            "freeze_fraction": self.freeze_fraction,
        }
