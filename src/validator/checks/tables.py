from typing import Dict, Hashable, Sequence

import pandas as pd

from ..check import register_check
from ..helpers import sort_partial

@register_check(name='tables_not_empty', message='Has no tables')
def not_empty(dfs: Dict[Hashable, pd.DataFrame]) -> bool:
  return len(dfs) > 0

@register_check(message='Missing required table')
def has_tables(dfs: Dict[Hashable, pd.DataFrame], *, tables: Sequence[Hashable], coerce: bool = False) -> Dict[Hashable, bool]:
  for table in tables:
    if table not in dfs:
      if coerce:
        # NOTE: Modifies dictionary in place
        dfs[table] = pd.DataFrame()
  return {table: table in dfs for table in tables}

@register_check(message='Table not one of {tables}')
def only_has_tables(dfs: Dict[Hashable, pd.DataFrame], *, tables: Sequence[Hashable], coerce: bool = False) -> Dict[Hashable, bool]:
  for table in dfs:
    if table not in tables:
      if coerce:
        # NOTE: Modifies dictionary in place
        del dfs[table]
  return {table: table in tables for table in dfs}

@register_check(message='Table does not follow order {tables}')
def has_sorted_tables(dfs: Dict[Hashable, pd.DataFrame], *, tables: Sequence[Hashable], coerce: bool = False) -> Dict[Hashable, bool]:
  ordered = sort_partial(list(dfs), order=tables)
  if coerce:
    temp = {table: dfs[table] for table in ordered}
    # NOTE: Modifies dictionary in place
    dfs.clear()
    dfs.update(temp)
  return {table: table == ordered[i] for i, table in enumerate(dfs)}
