from typing import Any, Dict, Hashable, List, Sequence, Union

import pandas as pd

from ..check import check
from ..targets import Column
from ..helpers import sort_partial

@check(message='Has no rows')
def not_empty(df: pd.DataFrame) -> bool:
  return not df.empty

@check(message='Missing required column', axis='column')
def has_columns(df: pd.DataFrame, *, columns: Sequence[Hashable], fill: bool = False, value: Any = pd.NA, dtype: str = None) -> Dict[Hashable, bool]:
  for column in columns:
    if column not in df and fill:
      # NOTE: Modifies dataframe in place
      df[column] = pd.Series(value, dtype=dtype, index=df.index)
  return {column: column in df for column in columns}

@check(message='Column not one of {columns}', axis='column')
def only_has_columns(df: pd.DataFrame, *, columns: Sequence[Hashable], drop: bool = False) -> Dict[Hashable, bool]:
  if drop:
    extras = [column for column in df if column not in columns]
    # NOTE: Modifies dataframe in place
    df.drop(columns=extras, inplace=True)
  return {column: column in columns for column in df}

@check(message='Column does not follow order {columns}', axis='column')
def has_sorted_columns(df: pd.DataFrame, *, columns: Sequence[Hashable], sort: bool = False) -> Dict[Hashable, bool]:
  """
  Check whether column names are sorted.

  Args:
    columns: Sorted column names.
    sort: Whether to sort table columns to match `columns`.
      Columns not named in `columns` are ignored and left in place.

  Examples:
    >>> df = pd.DataFrame(columns=['y', 'z', 'x'])
    >>> has_sorted_columns(df, columns=['x', 'z'])
    {'y': True, 'z': False, 'x': False}
    >>> has_sorted_columns(df, columns=['x', 'z'], sort=True)
    >>> list(df)
    ['y', 'x', 'z']
  """
  ordered = sort_partial(list(df), order=columns)
  if sort:
    # NOTE: Modifies dataframe in place
    for i, column in enumerate(ordered):
      s = df.pop(column)
      df.insert(i, column, s)
  return {column: column == ordered[i] for i, column in enumerate(df)}

@check(
  message='Duplicate combination of columns {columns}',
  requires=lambda columns: [Column(column) for column in columns]
)
def unique_rows(df: pd.DataFrame, *, columns: Sequence[Hashable] = None) -> pd.Series:
  """
  Check whether rows are unique.

  Args:
    columns: Names of columns to consider for testing uniqueness.
      By default, all columns are used.

  Example:
    >>> df = pd.DataFrame({'x': [1, 1, 1], 'y': [1, 2, 2]})
    >>> unique_rows(df, columns=['x'])
    0    False
    1    False
    2    False
    dtype: bool
    >>> unique_rows(df)
    0     True
    1    False
    2    False
    dtype: bool
  """
  return ~df.duplicated(subset=columns, keep=False)

@check(
  message='Not found in {list(columns.values)}',
  requires=lambda columns: [Column(column) for column in [*columns, *columns.values()]]
)
def in_columns(df: pd.DataFrame, *, columns: Dict[Hashable, Hashable]) -> pd.Series:
  """
  Check whether column values exist in other columns of the table.

  Rows with one or more null values in `columns` are ignored.
  """
  local = df[columns.keys()]
  local_key = pd.MultiIndex.from_frame(local)
  foreign_key = pd.MultiIndex.from_frame(df[columns.values()])
  # Pass check if one or more local columns are null
  return local_key.isin(foreign_key) | local.isnull().any(axis=1)

@check(
  message='Not found in {table}.{list(columns.values())}',
  requires=lambda table, columns: [Column(column) for column in columns] + [Column(column, table=table) for column in columns.values()]
)
def in_foreign_columns(
  df: pd.DataFrame,
  dfs: Dict[Hashable, pd.DataFrame],
  *,
  table: Hashable,
  columns: Dict[Hashable, Hashable]
) -> pd.Series:
  """
  Check whether rows exist in a foreign table.

  Rows with one or more null values in `columns` are ignored.

  Args:
    table: Foreign table name.
    columns: Column mapping, with column names for `df` as keys,
      and column names for `table` as values.

  Example:
    >>> df = pd.DataFrame({'x': [0, 0, 1, pd.NA], 'y': [0, 1, 1, 2]})
    >>> dfs = {'table': pd.DataFrame({'x': [0, 0], 'y': [0, 1]})}
    >>> in_foreign_columns(df, dfs, table='table', columns={'x': 'x', 'y': 'y'})
    0     True
    1     True
    2    False
    3     True
    dtype: bool
    >>> in_foreign_columns(df, dfs, table='table', columns={'y': 'y'})
    0     True
    1     True
    2     True
    3    False
    dtype: bool
  """
  local = df[columns.keys()]
  local_key = pd.MultiIndex.from_frame(local)
  foreign_key = pd.MultiIndex.from_frame(dfs[table][columns.values()])
  # Pass check if one or more local columns are null
  return local_key.isin(foreign_key) | local.isnull().any(axis=1)

@check(
  message='Columns {list(columns.keys())} do not match {table}.{list(columns.values())} when joined on {join}',
  requires=lambda table, join, columns: [Column(column) for column in [*join, *columns]] + [Column(column, table=table) for column in [*join.values(), *columns.values()]]
)
def matches_foreign_columns(
  df: pd.DataFrame,
  dfs: Dict[Hashable, pd.DataFrame],
  *,
  table: Hashable,
  join: Dict[Hashable, Hashable],
  columns: Dict[Hashable, Hashable]
) -> pd.Series:
  """
  Check whether rows match a foreign table following a join.

  Requires a many-to-one relationship with the foreign table.

  Args:
    table: Foreign table name.
    join: Columns to use for the join, with column names for `df` as keys,
      and column names for `table` as values.
    columns: Columns to match, with column names for `df` as keys,
      and column names for `table` as values.

  Example:
    >>> df = pd.DataFrame({'id': [0, 0, 1, 1], 'x': [0, 1, 1, pd.NA]})
    >>> dfs = {'table': pd.DataFrame({'rid': [0, 1], 'rx': [0, 1]})}
    >>> matches_foreign_columns(
    ...   df, dfs, table='table', join={'id': 'rid'}, columns={'x': 'rx'}
    ... )
    0     True
    1    False
    2     True
    3    False
    dtype: bool
  """
  local = df[[*join]]
  foreign = dfs[table][[*join.values(), *columns.values()]]
  joined = local.merge(
    foreign,
    how='left',
    left_on=list(join),
    right_on=list(join.values()),
    suffixes=[None, '.y'],
    validate='many_to_one',
    copy=False
  )
  ref_columns = [
    col if col in joined else f'{col}.y' for col in columns.values()
  ]
  refs = joined[ref_columns].rename(columns={v: k for k, v in columns.items()})
  return (df[[*columns]] == refs).all(axis=1)
