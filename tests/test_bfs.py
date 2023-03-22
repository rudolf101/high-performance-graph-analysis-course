import pathlib

import pytest

from project.bfs import bfs
from tests.utils import load_test_data_json

test_bfs_path = pathlib.Path(__file__).parent / "resources" / "test_bfs.json"


@pytest.mark.parametrize("test", load_test_data_json(test_bfs_path))
def test_bfs(test):
    matrix, source, ans = test
    assert bfs(matrix, source) == ans
