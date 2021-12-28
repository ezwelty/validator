from typing import Dict, Hashable, Sequence, Union

import pandas as pd

from ..check import check
from ..helpers import sort_partial

@check(message='Has no tables')
def not_empty(dfs: Dict[Hashable, pd.DataFrame]) -> bool:
  return len(dfs) > 0

@check(message='Missing required table', axis='table')
def has_tables(dfs: Dict[Hashable, pd.DataFrame], *, tables: Sequence[Hashable], fill: bool = False) -> Dict[Hashable, bool]:
  for table in tables:
    if table not in dfs:
      if fill:
        # NOTE: Modifies dictionary in place
        dfs[table] = pd.DataFrame()
  return {table: table in dfs for table in tables}

@check(message='Table not one of {tables}', axis='table')
def only_has_tables(dfs: Dict[Hashable, pd.DataFrame], *, tables: Sequence[Hashable], drop: bool = False) -> Dict[Hashable, bool]:
  for table in dfs:
    if table not in tables:
      if drop:
        # NOTE: Modifies dictionary in place
        del dfs[table]
  return {table: table in tables for table in dfs}

@check(message='Table does not follow order {tables}', axis='table')
def has_sorted_tables(dfs: Dict[Hashable, pd.DataFrame], *, tables: Sequence[Hashable], sort: bool = False) -> Dict[Hashable, bool]:
  ordered = sort_partial(list(dfs), order=tables)
  if sort:
    temp = {table: dfs[table] for table in ordered}
    # NOTE: Modifies dictionary in place
    dfs.clear()
    dfs.update(temp)
  return {table: table == ordered[i] for i, table in enumerate(dfs)}
