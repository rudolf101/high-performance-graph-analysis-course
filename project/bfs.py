from typing import List, Union

from pygraphblas import Vector, Matrix, INT64, BOOL
from pygraphblas.descriptor import RSC, S, C


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
    front = Vector.sparse(typ=BOOL, size=size)
    front[source] = True

    ans = Vector.sparse(typ=INT64, size=size)
    curr_depth = 0

    while front.nvals:
        ans.assign_scalar(value=curr_depth, mask=front)
        front.vxm(other=graph, out=front, mask=ans, desc=RSC)
        curr_depth += 1

    return list(ans.get(i, default=-1) for i in range(size))


def msbfs(graph: Matrix, sources: List[int]) -> List[List[Union[int, List[int]]]]:
    """
    Matrix-like BFS of a directed graph by multiple sources

    Parameters
    ----------
    graph: Matrix
        adjacency boolean matrix of a graph
    sources: List[int]
        start vertices
    Returns
    -------
    result: List[List[int, List[int]]]
        list of pairs: source, list of parents for each node, where source is -1 and unreachable nodes are -2
    """
    graph_size = graph.nrows
    sources_size = len(sources)

    ans = Matrix.sparse(typ=INT64, nrows=sources_size, ncols=graph_size)
    front = Matrix.sparse(typ=INT64, nrows=sources_size, ncols=graph_size)

    for row, source in enumerate(sources):
        front[row, source] = -1

    while front.nvals:
        ans.assign_matrix(value=front, mask=front, desc=S)
        front.apply(op=INT64.POSITIONJ, out=front)
        front.mxm(other=graph, semiring=INT64.MIN_FIRST, out=front, mask=ans, desc=RSC)

    ans.assign_scalar(value=-2, mask=ans.pattern(), desc=C)

    return [
        [source, list(ans.extract_row(row_index).vals)]
        for row_index, source in enumerate(sources)
    ]
