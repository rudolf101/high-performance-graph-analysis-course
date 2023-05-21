import boltons.queueutils
import heapq
import itertools
import math
import networkx as nx
from typing import Hashable


def dijkstra_sssp(graph: nx.Graph, start: Hashable) -> dict[Hashable, float]:
    """
    Finds the shortest paths from a single source vertex using the classical Dijkstra's algorithm.

    :param graph: The graph on which the algorithm is run. If the edges of the graph have a 'weight' attribute,
                  it will be used as the edge weight; otherwise, a weight of 1 will be assumed. The weights are
                  assumed to be non-negative.
    :param start: The source vertex from which the paths are calculated.

    :return: A dictionary containing the distances from the start vertex to each vertex in the graph. If a vertex
             is unreachable, its distance will be +inf.
    """
    distances = {node: math.inf for node in graph.nodes}
    distances[start] = 0

    queue = [(0, start)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor].get("weight", 1)
            new_distance = distances[current_node] + weight

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(queue, (new_distance, neighbor))

    return distances


class DynamicSSSP:
    """
    Dynamic Dijkstra-like algorithm for computing single-source shortest paths.
    """

    def __init__(self, graph: nx.DiGraph, start: Hashable):
        """
        Initializes the algorithm to run on the provided directed graph and start vertex.

        :param graph: The directed graph on which the algorithm is run. If the edges of the graph have a 'weight'
                      attribute, it will be used as the edge weight; otherwise, a weight of 1 will be assumed.
                      The weights are assumed to be non-negative.
        :param start: The source vertex from which the paths are calculated.
        """
        self._graph = graph
        self._start = start
        self._distances = dijkstra_sssp(graph, start)
        self._modified_nodes = set()

    def insert_edge(self, source: Hashable, target: Hashable, weight: float = 1):
        """
        Inserts or updates an edge in the graph.

        :param source: The source vertex of the edge.
        :param target: The target vertex of the edge.
        :param weight: The weight of the edge (default: 1).
        """
        self._graph.add_edge(source, target, weight=weight)
        self._modified_nodes.add(target)
        self._distances.setdefault(source, math.inf)
        self._distances.setdefault(target, math.inf)

    def delete_edge(self, source: Hashable, target: Hashable):
        """
        Deletes an edge from the graph.

        :param source: The source vertex of the edge.
        :param target: The target vertex of the edge.
        """
        self._graph.remove_edge(source, target)
        self._modified_nodes.add(target)

    def query_distances(self) -> dict[Hashable, float]:
        """
        Returns the distances from the start vertex to each vertex in the graph.

        :return: A dictionary containing the distances from the start vertex to each vertex in the graph. If a vertex
                 is unreachable, its distance will be +inf.
        """
        if self._modified_nodes:
            self._update_distances()
            self._modified_nodes.clear()
        return self._distances

    def _update_distances(self):
        """
        Applies the accumulated graph updates to the stored distances.
        """
        rhs = {}
        queue = boltons.queueutils.HeapPriorityQueue(priority_key=lambda x: x)

        for node in self._modified_nodes:
            rhs[node] = self._calculate_rhs(node)
            if rhs[node] != self._distances[node]:
                queue.add(node, priority=min(rhs[node], self._distances[node]))

        while queue:
            node = queue.pop()

            if rhs[node] < self._distances[node]:
                self._distances[node] = rhs[node]
                successors = self._graph.successors(node)
            else:
                self._distances[node] = math.inf
                successors = itertools.chain(self._graph.successors(node), [node])

            for successor in successors:
                rhs[successor] = self._calculate_rhs(successor)
                if rhs[successor] != self._distances[successor]:
                    queue.add(
                        successor,
                        priority=min(rhs[successor], self._distances[successor]),
                    )
                else:
                    if successor in queue._entry_map:
                        queue.remove(successor)

    def _calculate_rhs(self, vertex: Hashable):
        """
        Calculates the current right-hand-side (rhs) value for the provided vertex.

        :param vertex: The vertex for which to calculate the rhs value.

        :return: The calculated rhs value.
        """
        if vertex == self._start:
            return 0
        return min(
            (
                self._distances[predecessor]
                + self._graph[predecessor][vertex].get("weight", 1)
                for predecessor in self._graph.predecessors(vertex)
            ),
            default=math.inf,
        )
