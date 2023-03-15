import json
from pathlib import Path
from typing import List, Tuple, Union

from pygraphblas import Matrix, BOOL


def load_test_data_json(path: Path) -> List[Tuple[Matrix, int, List[int]]]:
    matrix, source, ans = [], [], []

    with open(path, 'r') as file:
        data_chunks = json.load(file)["data"]
        for data in data_chunks:
            matrix.append(Matrix.from_lists(*data["matrix"], V=True, typ=BOOL))
            source.append(data["source"])
            ans.append(data["ans"])

    return list(zip(matrix, source, ans))
