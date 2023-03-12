from typing import List

from pygraphblas import Vector, Matrix, INT64, BOOL


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
    visited = Vector.sparse(BOOL, size)
    front = Vector.sparse(BOOL, size)
    front[source] = True

    ans = Vector.dense(INT64, size, fill=-1)
    curr_depth = 0
    previous = None

    while previous != visited.nvals:
        previous = visited.nvals
        ans.assign_scalar(value=curr_depth, mask=front)
        visited |= front
        front = front.vxm(graph)
        front.assign_scalar(value=False, mask=visited)
        curr_depth += 1

    return list(ans.vals)
