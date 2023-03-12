import pytest
from project.bfs import *
from pygraphblas import Matrix


@pytest.mark.parametrize(
    "u, v, source, ans",
    [
        (
            [0, 1, 2, 3, 3, 4, 4, 4, 4, 5, 6, 7, 7, 8],
            [1, 2, 3, 3, 4, 5, 6, 6, 7, 7, 7, 7, 8, 8],
            1,
            [-1, 0, 1, 2, 3, 4, 4, 4, 5],
        ),
        (
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 0],
            2,
            [4, 5, 0, 1, 2, 3],
        ),
    ],
)
def test_bfs(u, v, source, ans):
    assert bfs(Matrix.from_lists(u, v, True, typ=BOOL), source) == ans
