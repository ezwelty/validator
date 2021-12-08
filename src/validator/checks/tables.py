from typing import Dict, Hashable, Sequence, Tuple

import pandas as pd

from ..check import check

@check(message='Does not have table {table}')
def has_table(dfs: Dict[Hashable, pd.DataFrame], *, table: Hashable) -> bool:
  return table in dfs

@check
def fill_table(dfs: Dict[Hashable, pd.DataFrame], *, table: Hashable, value: pd.DataFrame = pd.DataFrame()) -> Tuple[bool, Dict[Hashable, pd.DataFrame]]:
  if table not in dfs:
    # NOTE: Modifies tables in place
    dfs[table] = value
  return True, dfs

@check(message='Has no tables')
def not_empty(dfs: Dict[Hashable, pd.DataFrame]) -> bool:
  return len(dfs) > 0

# TODO: Error lists tables that do not belong
@check(message='Has tables other than {tables}')
def only_has_tables(dfs: Dict[Hashable, pd.DataFrame], *, tables: Sequence[Hashable], drop: bool = False) -> Tuple[bool, pd.DataFrame]:
  extras = [table for table in dfs if table not in tables]
  if extras and drop:
    dfs = {k: v for k, v in dfs.items() if k not in extras}
  if extras:
    return False, dfs
  return True, dfs
