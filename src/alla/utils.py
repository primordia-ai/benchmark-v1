from enum import Enum


class BudgetMode(Enum):
    SYNTH = "synthetic"  # Uses the synthetic, unitless costs
    WALL = "wall"
    CPU_HR = "cpu_hr"
