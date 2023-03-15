from typing import List

from pygraphblas import Vector, Matrix, INT64, BOOL
from pygraphblas.descriptor import RSC


def bfs(graph: Matrix, source: int) -> List[int]:
    """
    Matrix-like BFS of a directed graph by source vertex

    Parameters
    ----------
    graph: Matrix
        adjacency boolean matrix of a graph
    source: int
        start vertex
    Returns
    -------
    result: List[int]
        list with distance(or -1 if no path) from source to another vertexes
    """
    size = graph.nrows
    front = Vector.sparse(BOOL, size)
    front[source] = True

    ans = Vector.sparse(INT64, size)
    curr_depth = 0

    while front.nvals:
        ans.assign_scalar(value=curr_depth, mask=front)
        front.vxm(graph, out=front, mask=ans, desc=RSC)
        curr_depth += 1

    return list(ans.get(i, default=-1) for i in range(size))
