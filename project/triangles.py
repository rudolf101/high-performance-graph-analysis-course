from typing import List

from pygraphblas import Matrix, BOOL, INT64
from pygraphblas.descriptor import T0


def _assert_adj_matrix_of_undirected_graph(m: Matrix):
    assert m.square
    assert m.type == BOOL
    assert m.iseq(m.transpose())


def vertex_triangle_count(graph: Matrix) -> List[int]:
    """
    Counts the number of triangles for each vertex of undirected graph

    Parameters
    ----------
    graph: Matrix
        adjacency boolean matrix of undirected graph
    Returns
    -------
    result: List[int]
        count of triangles that contain vertex with index i
    """
    _assert_adj_matrix_of_undirected_graph(graph)
    triangles = graph.mxm(graph, cast=INT64, mask=graph).reduce_vector(desc=T0)
    return list(
        (
            triangles.dense(
                INT64, size=triangles.size, fill=triangles if triangles.nvals else 0
            )
            / 2
        ).vals
    )


def cohen_algorithm(graph: Matrix) -> int:
    """
    Cohen's algorithm which calculates number of triangles in undirected graph

    Parameters
    ----------
    graph: Matrix
        adjacency boolean matrix of undirected graph
    Returns
    -------
    result: int
        number of triangles
    """
    _assert_adj_matrix_of_undirected_graph(graph)
    return sum(graph.tril()(graph.triu(), cast=INT64, mask=graph).vals) // 2


def sandia_algorithm(graph: Matrix) -> int:
    """
    Sandia algorithm which calculates number of triangles in undirected graph
    Parameters
    ----------
    graph: Matrix
        adjacency boolean matrix of undirected graph
    Returns
    -------
    result: int
        number of triangles
    """
    _assert_adj_matrix_of_undirected_graph(graph)
    u = graph.triu()
    return sum(u.mxm(u, cast=INT64, mask=u).vals)
