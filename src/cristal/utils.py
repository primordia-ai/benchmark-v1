from enum import Enum

import pydoc


class BudgetMode(Enum):
    SYNTH = "synthetic"  # Uses the synthetic, unitless costs
    WALL = "wall"
    CPU_HR = "cpu_hr"


def describe_class(cls) -> str:
    """Use pydoc to return a detailed, nicely formatted description of a class and its methods."""
    return pydoc.render_doc(cls)
