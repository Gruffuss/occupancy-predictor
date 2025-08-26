"""Basic test to ensure test infrastructure works."""

import occupancy


def test_version():
    """Test that version is defined."""
    assert occupancy.__version__ == "0.1.0"


def test_basic():
    """Basic test that always passes."""
    assert True
