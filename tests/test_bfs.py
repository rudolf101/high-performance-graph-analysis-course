import pathlib

import pytest

from project.bfs import bfs, msbfs
from tests.utils import load_test_data_json

test_bfs_path = pathlib.Path(__file__).parent / "resources" / "test_bfs.json"
test_msbfs_path = pathlib.Path(__file__).parent / "resources" / "test_msbfs.json"


@pytest.mark.parametrize("test", load_test_data_json(test_bfs_path))
def test_bfs(test):
    matrix, source, ans = test
    assert bfs(matrix, source) == ans


@pytest.mark.parametrize("test", load_test_data_json(test_msbfs_path))
def test_msbfs(test):
    matrix, sources, ans = test
    assert msbfs(matrix, sources) == ans
