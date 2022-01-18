from typing import List

import daff
import numpy as np
import pandas as pd


DAFF_TEMPLATE_PREFIX = """
<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<style TYPE='text/css'>
.highlighter .add {
  background-color: #7fff7f;
}

.highlighter .remove {
  background-color: #ff7f7f;
}

.highlighter td.modify {
  background-color: #7f7fff;
}

.highlighter td.conflict {
  background-color: #f00;
}

.highlighter .spec {
  background-color: #aaa;
}

.highlighter .move {
  background-color: #ffa;
}

.highlighter .null {
  color: #888;
}

.highlighter table {
  border-collapse:collapse;
}

.highlighter td, .highlighter th {
  border: 1px solid #2D4068;
  padding: 3px 7px 2px;
}

.highlighter th, .highlighter .header, .highlighter .meta {
  background-color: #aaf;
  font-weight: bold;
  padding-bottom: 4px;
  padding-top: 5px;
  text-align:left;
}

.highlighter tr.header th {
  border-bottom: 2px solid black;
}

.highlighter tr.index td, .highlighter .index, .highlighter tr.header th.index {
  background-color: white;
  border: none;
}

.highlighter .gap {
  color: #888;
}

.highlighter td {
  empty-cells: show;
  white-space: pre-wrap;
}
</style>
</head>
<body>
<div class='highlighter'>
"""

DAFF_TEMPLATE_SUFFIX = """
</div>
</body>
</html>
"""

def dataframe_to_string_list(
  df: pd.DataFrame, na: str = '', index: bool = True
) -> List[List[str]]:
  header = df.columns.astype('string').fillna(na).to_numpy()
  data = df.astype('string').fillna(na).to_numpy()
  if index:
    index_name = str(df.index.name or na)
    index_values = df.index.astype('string').fillna(na).to_numpy()
    header = np.concatenate(([index_name], header))
    data = np.column_stack((index_values, data))
  return np.row_stack((header, data)).tolist()

def table_diff(a: pd.DataFrame, b: pd.DataFrame, path: str) -> None:
  diff = daff.diff(
    dataframe_to_string_list(a),
    dataframe_to_string_list(b),
    flags=daff.CompareFlags()
  )
  render = daff.DiffRender().render(diff)
  with open(path, 'w') as file:
    file.write(DAFF_TEMPLATE_PREFIX + render.html() + DAFF_TEMPLATE_SUFFIX)
