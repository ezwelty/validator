from typing import Dict, Hashable, Sequence, Tuple

import pandas as pd

from ..check import register_check
from ..targets import Column
from ..helpers import sort_partial

@register_check(name='table_not_empty', message='Has no rows')
def not_empty(df: pd.DataFrame) -> bool:
  return not df.empty

@register_check(message='Missing required column', axis='column')
def has_columns(df: pd.DataFrame, *, columns: Sequence[Hashable], fill: bool = False) -> Dict[Hashable, bool]:
  for column in columns:
    if column not in df and fill:
      # NOTE: Modifies dataframe in place
      df[column] = pd.Series(pd.NA, dtype=object, index=df.index)
  return {column: column in df for column in columns}

@register_check(message='Column not one of {columns}', axis='column')
def only_has_columns(df: pd.DataFrame, *, columns: Sequence[Hashable], drop: bool = False) -> Dict[Hashable, bool]:
  if drop:
    extras = [column for column in df if column not in columns]
    # NOTE: Modifies dataframe in place
    df.drop(columns=extras, inplace=True)
  return {column: column in columns for column in df}

@register_check(message='Column does not follow order {columns}', axis='column')
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
    {'y': True, 'x': True, 'z': True}
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

@register_check(
  message='Duplicate combination of columns {columns}',
  required=lambda columns: [Column(column) for column in columns]
)
def unique_rows(df: pd.DataFrame, *, columns: Sequence[Hashable] = None) -> pd.Series:
  """
  Check whether rows are unique.

  Rows with null values are ignored.

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

    >>> df = pd.DataFrame({'x': [1, pd.NA, pd.NA], 'y': [1, pd.NA, pd.NA]})
    >>> unique_rows(df)
    0     True
    dtype: bool
  """
  if columns is not None:
    df = df[columns]
  return ~df.dropna().duplicated(keep=False)

@register_check(
  message='Not found in {list(columns.values())}',
  required=lambda columns: [Column(column) for column in [*columns, *columns.values()]]
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

@register_check(
  message='Not found in {table}.{list(columns.values())}',
  required=lambda table, columns: (
    [Column(column) for column in columns] +
    [Column(column, table=table) for column in columns.values()]
  )
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
    dtype: bool
    >>> in_foreign_columns(df, dfs, table='table', columns={'y': 'y'})
    0     True
    1     True
    2     True
    3    False
    dtype: bool
  """
  local = df[columns.keys()].dropna()
  local_key = pd.MultiIndex.from_frame(local)
  foreign_key = pd.MultiIndex.from_frame(dfs[table][columns.values()].dropna())
  return pd.Series(local_key.isin(foreign_key), index=local.index)

@register_check(
  message='Columns {list(columns.keys())} do not match {table}.{list(columns.values())} when joined on {join}',
  required=lambda table, join, columns: (
    [Column(column) for column in [*join, *columns]] +
    [Column(column, table=table) for column in [*join.values(), *columns.values()]]
  )
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

  Rows in either table with one or more nulls in a join or match column are
  ignored.

  Args:
    table: Foreign table name.
    join: Columns to use for the join, with column names for `df` as keys,
      and column names for `table` as values.
    columns: Columns to match, with column names for `df` as keys,
      and column names for `table` as values.

  Example:
    >>> df = pd.DataFrame(
    ...   {'id': [0, 0, 1, 1], 'x': [0, 1, 1, pd.NA]},
    ...   index=[0, 1, 3, 4]
    ... )
    >>> dfs = {'table': pd.DataFrame({'id': [0, 1], 'x': [0, 1]})}
    >>> valid = matches_foreign_columns(
    ...   df, dfs, table='table', join={'id': 'id'}, columns={'x': 'x'}
    ... )
    >>> df.loc[valid.index[~valid]]
       id  x
    1   0  1

    >>> dfs = {'table': pd.DataFrame({'id': [2], 'x': [0]})}
    >>> matches_foreign_columns(
    ...   df, dfs, table='table', join={'id': 'id'}, columns={'x': 'x'}
    ... )
    Series([], dtype: bool)
  """
  # Ignore join keys or lookup columns with null
  local = df[[*join, *columns]].dropna()
  foreign = dfs[table][[*join.values(), *columns.values()]].dropna()
  # Track left row numbers
  local['__row__'] = range(len(local))
  joined = local.merge(
    foreign,
    how='inner',
    left_on=list(join),
    right_on=list(join.values()),
    suffixes=['.x', '.y'],
    copy=False
  )
  local_cols = [f'{col}.x' if col not in joined else col for col in columns]
  foreign_cols = [
    f'{col}.y' if col not in joined else col for col in columns.values()
  ]
  local_to_foreign = dict(zip(local_cols, foreign_cols))
  # In case of many-to-many, require all are equal
  valid = (
    joined
    .groupby('__row__', sort=False)
    .apply(
      lambda g: (
        g[local_cols].rename(columns=local_to_foreign) == g[foreign_cols]
      ).all(None)
    )
  )
  if isinstance(valid, pd.DataFrame):
    # groupby-apply returns empty frame if join is empty
    # Ensure integer index to avoid IndexError
    # 'arrays used as indices must be of integer or boolean type'
    valid = pd.Series(dtype=bool, index=pd.Index([], dtype=int))
  # Restore original index
  valid.index = local.index[valid.index]
  return valid

@register_check(message='Column with new name already exists', axis='column')
def rename_columns(
  df: pd.DataFrame, *, columns: Dict[Hashable, Hashable]
) -> Tuple[Dict[Hashable, bool], pd.DataFrame]:
  """
  Rename selected columns, unless columns with new names already exist.

  Examples
  --------
  >>> df = pd.DataFrame(columns=['x', 'y', 'z'])
  >>> valid, output = rename_columns(df, columns={'x': 'xx', 'y': 'z'})
  >>> valid
  {'x': True, 'y': False}
  >>> list(output)
  ['xx', 'y', 'z']
  """
  valid = {
    name: name not in df or rename not in df
    for name, rename in columns.items()
  }
  renames = {name: rename for name, rename in columns.items() if valid[name]}
  return valid, df.rename(columns=renames)

@register_check(message='Column contains only null values', axis='column')
def columns_not_null(
  df: pd.DataFrame, *, drop: bool = False
) -> Tuple[pd.Series, pd.Series]:
  """
  Check that columns are not null.

  Examples
  --------
  >>> df = pd.DataFrame({'x': [0, pd.NA], 'y': [pd.NA, pd.NA]})
  >>> valid, output = columns_not_null(df)
  >>> valid
  x     True
  y    False
  dtype: bool
  >>> output is df
  True
  >>> valid, output = columns_not_null(df, drop=True)
  >>> valid
  x    True
  dtype: bool
  >>> list(output)
  ['x']
  """
  valid = df.notnull().any(axis='index')
  if drop:
    # Use drop to ensure a copy (and not a view) is returned.
    df = df.drop(columns=valid[~valid].index)
    valid = valid[valid]
  return valid, df

@register_check(message='Row contains only null values')
def rows_not_null(
  df: pd.DataFrame, *, drop: bool = False
) -> Tuple[pd.Series, pd.DataFrame]:
  """
  Check that rows are not null.

  Examples
  --------
  >>> df = pd.DataFrame({'x': [0, pd.NA], 'y': [pd.NA, pd.NA]})
  >>> valid, output = rows_not_null(df)
  >>> valid
  0     True
  1    False
  dtype: bool
  >>> output is df
  True
  >>> valid, output = rows_not_null(df, drop=True)
  >>> valid
  0    True
  dtype: bool
  >>> list(output.index)
  [0]
  """
  valid = df.notnull().any(axis='columns')
  if drop:
    # Use drop to ensure a copy (and not a view) is returned.
    df = df.drop(index=valid[~valid].index)
    valid = valid[valid]
  return valid, df

@register_check(message='Column contains non-null values', axis='column')
def columns_missing_or_null(
  df: pd.DataFrame, *, columns: Sequence[Hashable]
) -> Dict[Hashable, bool]:
  """
  Check that selected columns are missing or null.

  Examples
  --------
  >>> df = pd.DataFrame({'x': [0, pd.NA], 'y': [pd.NA, pd.NA]})
  >>> columns_missing_or_null(df, columns=['x', 'y'])
  {'x': False, 'y': True}
  """
  return {
    column: column not in df or not df[column].notnull().any()
    for column in columns
  }
