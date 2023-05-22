import math
import networkx as nx
from pytest import approx

from project.sssp_dynamic import dijkstra_sssp, DynamicSSSP


def test_dijkstra_sssp_unreachable_vertex():
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=1)
    graph.add_edge("C", "D", weight=1)
    start = "A"
    distances = dijkstra_sssp(graph, start)
    assert distances == {"A": 0, "B": 1, "C": math.inf, "D": math.inf}


def test_dijkstra_sssp():
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=1)
    graph.add_edge("B", "C", weight=2)
    graph.add_edge("A", "C", weight=4)
    start = "A"
    distances = dijkstra_sssp(graph, start)
    assert distances == {"A": 0, "B": 1, "C": 3}


def test_dynamic_sssp_insert_edge():
    graph = nx.DiGraph()
    start = "A"
    graph.add_edge("A", "B", weight=2)

    dynamic_sssp = DynamicSSSP(graph, start)
    dynamic_sssp.insert_edge("B", "C", weight=3)

    distances = dynamic_sssp.query_distances()
    assert distances == {"A": 0, "B": 2, "C": 5}


def test_dynamic_sssp_delete_edge():
    graph = nx.DiGraph()
    start = "A"
    graph.add_edge("A", "B", weight=2)

    dynamic_sssp = DynamicSSSP(graph, start)
    dynamic_sssp.insert_edge("B", "C", weight=3)
    dynamic_sssp.delete_edge("B", "C")

    distances = dynamic_sssp.query_distances()
    assert distances == {"A": 0, "B": 2, "C": math.inf}


def test_dynamic_sssp_update_distances():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=2)
    graph.add_edge("B", "C", weight=3)
    start = "A"
    dynamic_sssp = DynamicSSSP(graph, start)

    dynamic_sssp.insert_edge("A", "C", weight=5)

    distances = dynamic_sssp.query_distances()
    assert distances == {"A": 0, "B": 2, "C": 5}


def test_dynamic_sssp_query_distances_multiple_updates():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=1)
    graph.add_edge("B", "C", weight=2)
    graph.add_edge("A", "C", weight=4)
    start = "A"
    dynamic_sssp = DynamicSSSP(graph, start)

    dynamic_sssp.insert_edge("B", "D", weight=2)
    dynamic_sssp.insert_edge("D", "C", weight=1)
    dynamic_sssp.delete_edge("B", "C")

    distances = dynamic_sssp.query_distances()
    assert distances == {"A": 0, "B": 1, "C": 4, "D": 3}
