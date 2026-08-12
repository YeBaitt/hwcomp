import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False, help="运行需要 GPU 的测试")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="需要 GPU")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
