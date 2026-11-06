"""Categorical simulation training profile."""
from dataclasses import dataclass

@dataclass(frozen=True)
class PolicyProfile:
    policy_kind: str
    indicator_temperature: float
    ordered_logit_scale: float = 1.0
    validation_matching: str = "hard"
    dro_evaluation_matching: str = "hard"
    checkpoint: str = "best"
    restore_best_alpha: bool = True
    minimum_nonrobust_epochs: int = 100
    propensity_model: str = "simulation"
    alpha_bound_order: str = "divide"

SIMULATION_PROFILE = PolicyProfile(policy_kind="categorical", indicator_temperature=0.05)
