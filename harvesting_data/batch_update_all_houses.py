"""
DEPRECATED: Use 'python -m harvesting_data.cli' instead.

This file is kept for backwards compatibility.
"""
from .cli import main, update_all_sequential
from .storage import load_house_tokens

# Legacy function
def update_all_houses():
    """Legacy function - use main() instead."""
    houses = load_house_tokens("small_mishkit.csv")
    update_all_sequential(houses)


if __name__ == "__main__":
    main()
