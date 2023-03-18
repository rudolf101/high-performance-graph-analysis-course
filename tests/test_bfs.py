import pathlib

from project.bfs import bfs
from tests.utils import load_test_data_json

test_bfs_path = pathlib.Path(__file__).parent / "resources" / "test_bfs.json"

tests = load_test_data_json(test_bfs_path)


# TODO: Make it parametrized
def test_bfs0():
    matrix, source, ans = tests[0]
    assert bfs(matrix, source) == ans


def test_bfs1():
    matrix, source, ans = tests[1]
    assert bfs(matrix, source) == ans
