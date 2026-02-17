"""
TrimCI Unified Logging System

Verbosity levels:
  0 - Silent: no stdout output
  1 - Normal: important messages only (log_important)
  2 - Verbose: all messages (log_important + log_verbose)

File logs (summary_logger, etc.) always write, unaffected by verbosity.
"""

import logging
import sys

# Global verbosity state
_verbosity = 1

# Color constants
BLUE = "\033[94m"
RESET = "\033[0m"


def setup_logging(verbosity: int = 1):
    """
    Set global verbosity level and configure logging.
    
    Args:
        verbosity: 0=silent, 1=normal, 2=verbose
    """
    global _verbosity
    _verbosity = verbosity
    
    # Clear existing handlers
    logging.getLogger().handlers.clear()
    
    # Configure logging level
    if verbosity >= 2:
        level = logging.INFO
    elif verbosity >= 1:
        level = logging.WARNING
    else:
        level = logging.ERROR  # Silent mode
    
    # Custom formatter with blue color and timestamp
    class ColorFormatter(logging.Formatter):
        def format(self, record):
            record.asctime = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
            formatted = f"{record.asctime}: {record.getMessage()}"
            return f"{BLUE}{formatted}{RESET}"
    
    logging.basicConfig(level=level, stream=sys.stdout, force=True)
    for handler in logging.root.handlers:
        handler.setFormatter(ColorFormatter())


def log_important(msg: str, prefix: str = ""):
    """
    Log important messages - shown when verbosity >= 1.
    
    Args:
        msg: Log message
        prefix: Optional prefix (e.g., "[OrbLab] ")
    """
    if _verbosity >= 1:
        full_msg = f"{prefix}{msg}" if prefix else msg
        logging.warning(full_msg)


def log_verbose(msg: str, prefix: str = ""):
    """
    Log verbose messages - shown when verbosity >= 2.
    
    Args:
        msg: Log message
        prefix: Optional prefix (e.g., "[OrbLab] ")
    """
    if _verbosity >= 2:
        full_msg = f"{prefix}{msg}" if prefix else msg
        logging.info(full_msg)


def get_verbosity() -> int:
    """Get current verbosity level."""
    return _verbosity


def set_verbosity(verbosity: int):
    """Set verbosity level directly (without reconfiguring logging)."""
    global _verbosity
    _verbosity = verbosity


def get_verbosity_from_args(args) -> int:
    """
    Extract verbosity level from args object.
    
    Priority: args.verbosity > args.verbose (bool conversion)
    
    Args:
        args: Namespace or object with verbosity/verbose attributes
        
    Returns:
        Verbosity level (0, 1, or 2)
    """
    if hasattr(args, 'verbosity') and args.verbosity is not None:
        return int(args.verbosity)
    elif hasattr(args, 'verbose'):
        return 1 if args.verbose else 0
    return 1  # Default
