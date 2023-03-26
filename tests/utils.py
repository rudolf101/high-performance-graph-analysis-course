import json
from pathlib import Path
from typing import List, Tuple, Union

from pygraphblas import Matrix, BOOL


def load_test_data_json(
    path: Path,
) -> List[Tuple[Matrix, Union[int, List[int]], Union[int, List[int]]]]:
    matrix, source, ans = [], [], []

    with open(path, "r") as file:
        data_chunks = json.load(file)["data"]
        for data in data_chunks:
            l1, l2 = data["matrix"]
            n = max(l1 + l2) + 1
            matrix.append(Matrix.from_lists(l1, l2, V=True, typ=BOOL, nrows=n, ncols=n))
            source.append(data["source"])
            ans.append(data["ans"])

    return list(zip(matrix, source, ans))
