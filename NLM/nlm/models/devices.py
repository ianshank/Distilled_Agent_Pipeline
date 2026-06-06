"""
Device selection utilities.

Centralizes the CUDA > MPS > CPU selection logic that was previously duplicated
in the inference server, so training, inference, and evaluation all agree on how
a device is chosen. The preference order is data-driven (a list of device-type
strings) rather than hardcoded at each call site.
"""

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

# Default device preference order. Callers (e.g. TrainingConfig.device_preference)
# may override this; nothing downstream should hardcode an order of its own.
DEFAULT_DEVICE_PREFERENCE: List[str] = ["cuda", "mps", "cpu"]


def _is_available(device_type: str) -> bool:
    """Return whether a device type is usable in the current environment."""
    if device_type == "cuda":
        return torch.cuda.is_available()
    if device_type == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if device_type == "cpu":
        return True
    logger.warning("Unknown device type '%s' in preference list; ignoring", device_type)
    return False


def select_device(preference: Optional[List[str]] = None) -> torch.device:
    """
    Select the best available device honoring a preference order.

    Args:
        preference: Ordered device-type strings (subset of cuda/mps/cpu). When
            None, :data:`DEFAULT_DEVICE_PREFERENCE` is used.

    Returns:
        The first available :class:`torch.device` from the preference order,
        falling back to CPU (always available).
    """
    pref = preference or DEFAULT_DEVICE_PREFERENCE

    for device_type in pref:
        if _is_available(device_type):
            device = torch.device(device_type)
            if device_type == "cuda":
                try:
                    logger.info("Selected CUDA device: %s", torch.cuda.get_device_name(0))
                except Exception:  # pragma: no cover - defensive, name lookup is cosmetic
                    logger.info("Selected CUDA device")
            else:
                logger.info("Selected device: %s", device_type)
            return device

    logger.warning("No preferred device available from %s; falling back to CPU", pref)
    return torch.device("cpu")
