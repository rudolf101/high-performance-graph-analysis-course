import pathlib

import pytest

from project.triangles import vertex_triangle_count
from tests.utils import load_test_data_json

test_triangles_path = (
    pathlib.Path(__file__).parent / "resources" / "test_triangles.json"
)


@pytest.mark.parametrize("test", load_test_data_json(test_triangles_path))
def test_triangles(test):
    matrix, source, ans = test
    a = vertex_triangle_count(matrix)
    assert a == ans
