import pytest

np = pytest.importorskip("numpy")

from drl_system.config import SafeFilterConfig
from drl_system.safety.safe_filter import SafeActionsFilter


def test_safe_filter_clamps_and_blocks():
    config = SafeFilterConfig(min_action=-0.5, max_action=0.5, forbidden_zones=[(-0.1, 0.1)])
    filter_ = SafeActionsFilter(config)
    unsafe = np.array([0.0])
    filtered = filter_.filter(unsafe)
    assert np.all(filtered == 0.0)
    high = np.array([1.0])
    filtered_high = filter_.filter(high)
    assert float(filtered_high[0]) == 0.5
