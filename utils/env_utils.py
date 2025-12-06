import sys


def is_colab() -> bool:
    """Return True if running inside Google Colab."""
    return "google.colab" in sys.modules
