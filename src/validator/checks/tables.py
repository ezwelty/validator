from typing import Dict, Hashable, Sequence, Tuple

import pandas as pd

from ..check import register_check
from ..helpers import sort_partial


@register_check(name='tables_not_empty', message='Has no tables')
def not_empty(dfs: Dict[Hashable, pd.DataFrame]) -> bool:
  return len(dfs) > 0

@register_check(message='Missing required table')
def has_tables(dfs: Dict[Hashable, pd.DataFrame], *, tables: Sequence[Hashable], fill: bool = False) -> Dict[Hashable, bool]:
  for table in tables:
    if table not in dfs:
      if fill:
        # NOTE: Modifies dictionary in place
        dfs[table] = pd.DataFrame()
  return {table: table in dfs for table in tables}

@register_check(message='Table not one of {tables}')
def only_has_tables(dfs: Dict[Hashable, pd.DataFrame], *, tables: Sequence[Hashable], drop: bool = False) -> Dict[Hashable, bool]:
  if drop:
    dropped = [table for table in dfs if table not in tables]
    for table in dropped:
      # NOTE: Modifies dictionary in place
      del dfs[table]
  return {table: table in tables for table in dfs}

@register_check(message='Table does not follow order {tables}')
def has_sorted_tables(dfs: Dict[Hashable, pd.DataFrame], *, tables: Sequence[Hashable], sort: bool = False) -> Dict[Hashable, bool]:
  ordered = sort_partial(list(dfs), order=tables)
  if sort:
    temp = {table: dfs[table] for table in ordered}
    # NOTE: Modifies dictionary in place
    dfs.clear()
    dfs.update(temp)
  return {table: table == ordered[i] for i, table in enumerate(dfs)}

@register_check(message='Table with new name already exists')
def rename_tables(
  dfs: Dict[Hashable, pd.DataFrame], *, tables: Dict[Hashable, Hashable]
) -> Tuple[Dict[Hashable, bool], Dict[Hashable, pd.DataFrame]]:
  """
  Rename selected tables, unless tables with new names already exist.

  Examples
  --------
  >>> dfs = {'x': pd.DataFrame(), 'y': pd.DataFrame(), 'z': pd.DataFrame()}
  >>> valid, output = rename_tables(dfs, tables={'x': 'xx', 'y': 'z'})
  >>> valid
  {'x': True, 'y': False}
  >>> list(output)
  ['xx', 'y', 'z']
  """
  valid = {
    name: name not in dfs or rename not in dfs
    for name, rename in tables.items()
  }
  renamed = {
    tables.get(name, name) if valid.get(name, False) else name: dfs[name]
    for name in dfs
  }
  return valid, renamed
