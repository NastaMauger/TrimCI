"""TrimCI runner subpackage.

Provides high-level helpers to run TrimCI calculations
from FCIDUMP or molecule definitions.
"""

from .trimci_driver import run_full_calculation
from .run_trimci import main as cli_main
from .parallel_runner import run_trimci_main_calculation_parallel

__all__ = [
    "run_full_calculation",
    "run_trimci_main_calculation_parallel",
    "run_auto",
    "cli_main",
    "read_fcidump",
]