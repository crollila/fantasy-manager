"""Keep the suite hermetic: the background refresh loop now runs at startup."""
import os
import pytest


@pytest.fixture(autouse=True)
def no_background_refresh(monkeypatch):
    """Tests must not reach the network. Individual tests opt in by clearing this."""
    monkeypatch.setenv('FANTASY_DISABLE_AUTO_REFRESH', '1')
    yield


def pytest_configure(config):
    os.environ.setdefault('FANTASY_DISABLE_AUTO_REFRESH', '1')
