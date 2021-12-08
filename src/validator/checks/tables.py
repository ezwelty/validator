from typing import Dict, Hashable, Sequence, Union

import pandas as pd

from ..check import check

@check(message='Has no tables')
def not_empty(dfs: Dict[Hashable, pd.DataFrame]) -> bool:
  return len(dfs) > 0

@check(message='Missing required table', axis='table')
def has_tables(dfs: Dict[Hashable, pd.DataFrame], *, tables: Sequence[Hashable], fill: bool = False, value: pd.DataFrame = pd.DataFrame) -> Dict[Hashable, bool]:
  for table in tables:
    if table not in dfs:
      if fill:
        # NOTE: Modifies dataframes in place
        dfs[table] = value
  return {table: table in dfs for table in tables}

@check(message='Table not one of {tables}', axis='table')
def only_has_tables(dfs: Dict[Hashable, pd.DataFrame], *, tables: Sequence[Hashable], drop: bool = False) -> Dict[Hashable, bool]:
  for table in dfs:
    if table not in tables:
      if drop:
        # NOTE: Modifies dataframes in place
        del dfs[table]
  return {table: table in tables for table in dfs}
