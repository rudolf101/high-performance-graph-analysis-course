import pytest

from project.path_finder import *
from tests.utils import matrix_with_weight_from_edge_list


@pytest.mark.parametrize(
    "edge_list, ans",
    [
        (
            [(0, 1.0, 1), (1, 2.0, 2)],
            [
                (0, [0.0, 1.0, 3.0]),
                (1, [math.inf, 0.0, 2.0]),
                (2, [math.inf, math.inf, 0.0]),
            ],
        ),
        (
            [(0, 1.0, 1), (0, 2.0, 2), (1, 3.0, 3), (2, 4.0, 3)],
            [
                (0, [0.0, 1.0, 2.0, 4.0]),
                (1, [math.inf, 0.0, math.inf, 3.0]),
                (2, [math.inf, math.inf, 0.0, 4.0]),
                (3, [math.inf, math.inf, math.inf, 0.0]),
            ],
        ),
    ],
)
def test_apsp(edge_list, ans):
    graph = matrix_with_weight_from_edge_list(edge_list)
    assert apsp(graph) == ans


@pytest.mark.parametrize(
    "edge_list, start, ans",
    [
        (
            [(0, 1.0, 1), (1, 1.0, 2)],
            [0, 1],
            [(0, [0.0, 1.0, 2.0]), (1, [math.inf, 0.0, 1.0])],
        ),
        (
            [(0, 3.0, 1), (0, 20.0, 2), (1, 4.0, 3), (2, 3.0, 3)],
            [0, 2],
            [(0, [0.0, 3.0, 20.0, 7.0]), (2, [math.inf, math.inf, 0.0, 3.0])],
        ),
    ],
)
def test_mssp(edge_list, start, ans):
    graph = matrix_with_weight_from_edge_list(edge_list)
    assert mssp(graph, start) == ans


@pytest.mark.parametrize(
    "edge_list, start, ans",
    [
        (
            [(0, 1.0, 1), (1, 1.0, 2)],
            1,
            [math.inf, 0.0, 1.0],
        ),
        (
            [(0, 2.4, 1), (1, 3.0, 3), (2, 45.3, 1), (3, 0.2, 0)],
            0,
            [0.0, 2.4, math.inf, 5.4],
        ),
    ],
)
def test_sssp(edge_list, start, ans):
    graph = matrix_with_weight_from_edge_list(edge_list)
    assert sssp(graph, start) == ans
