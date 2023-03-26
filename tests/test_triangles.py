import pathlib

import pytest

from project.triangles import vertex_triangle_count, cohen_algorithm, sandia_algorithm
from tests.utils import load_test_data_json

test_vertex_count_path = (
    pathlib.Path(__file__).parent / "resources" / "test_triangles_vertex_count.json"
)

test_triangles_algorithms_path = (
    pathlib.Path(__file__).parent / "resources" / "test_triangles_algorithms.json"
)


@pytest.mark.parametrize("test", load_test_data_json(test_vertex_count_path))
def test_vertex_count(test):
    matrix, source, ans = test
    assert vertex_triangle_count(matrix) == ans


@pytest.mark.parametrize("test", load_test_data_json(test_triangles_algorithms_path))
def test_triangles_algorithms(test):
    matrix, source, ans = test
    assert (cohen_algorithm(matrix), sandia_algorithm(matrix)) == (ans, ans)
