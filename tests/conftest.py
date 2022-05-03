from pathlib import Path

TEST_DIR = Path(__file__).parent

import os
import pytest

# Hack to make pytest stop on uncaught exceptions.
# See https://stackoverflow.com/a/62563106/2135504
if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value
