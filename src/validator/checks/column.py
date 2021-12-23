from typing import Any, Callable, Dict, Hashable, Iterable, List, Tuple

import pandas as pd

from ..check import check
from ..targets import Column


# ---- Checks ----

@check(message='Required value is missing')
def not_null(s: pd.Series) -> pd.Series:
  """Check whether values are not null."""
  return s.notnull()

@check(message='Duplicate value')
def unique(s: pd.Series) -> pd.Series:
  """Check whether values are unique."""
  return ~s.duplicated(keep=False)

@check(message='Value not in {values}')
def in_list(s: pd.Series, *, values: Iterable) -> pd.Series:
  """
  Check whether values are in a set of allowed values.

  Examples:
    >>> s = pd.Series([1, 2, None], dtype='float')
    >>> in_list(s, values=[1])
    0     True
    1    False
    2     <NA>
    dtype: boolean
    >>> s = pd.Series([1, 2, None], dtype='Int64')
    >>> in_list(s, values=[1])
    0     True
    1    False
    2     <NA>
    dtype: boolean
  """
  valid = pd.Series(dtype='boolean', index=s.index)
  not_null = s.notnull()
  valid[not_null] = s[not_null].isin(values)
  return valid

@check(message='Value < {min}')
def greater_than_or_equal_to(s: pd.Series, *, min: Any) -> pd.Series:
  """Check whether values are greater than or equal to a minimum."""
  return s.ge(min)

@check(message='Value > {max}')
def less_than_or_equal_to(s: pd.Series, *, max: Any) -> pd.Series:
  """Check whether values are less than or equal to a maximum."""
  return s.le(max)

@check(message='Length < {min}')
def length_greater_than_or_equal_to(s: pd.Series, *, min: int) -> pd.Series:
  """Check whether value lengths are greater than or equal to a minimum."""
  # Supports sequences (string, tuple, list) and collections (dictionary)
  # Conversion to Int64 needed from object input to ensure nulls are ignored
  return s.str.len().astype('Int64', copy=False).ge(min)

@check(message='Length > {max}')
def length_less_than_or_equal_to(s: pd.Series, *, max: int) -> pd.Series:
  """Check whether value lengths are less than or equal to a maximum."""
  # Supports sequences (string, tuple, list) and collections (dictionary)
  # Conversion to Int64 needed from object input to ensure nulls are ignored
  return s.str.len().astype('Int64', copy=False).le(max)

@check(message='Value does not match regular expression {regex}')
def matches_regex(s: pd.Series, *, regex: str) -> pd.Series:
  """Check whether values match a regular expression."""
  return s.str.fullmatch(regex).astype('boolean')

@check(
  message='Not found in {column}',
  requires=lambda column: Column(column)
)
def in_column(s: pd.Series, df: pd.DataFrame, *, column: Hashable) -> pd.Series:
  """Check whether values exist in another column."""
  other = df[column]
  return s.isin(other) | s.isnull()

@check(
  message='Not found in {table}.{column}',
  requires=lambda table, column: Column(column, table=table)
)
def in_foreign_column(
  s: pd.Series,
  dfs: Dict[Hashable, pd.DataFrame],
  *,
  table: Hashable,
  column: Hashable
) -> pd.Series:
  """
  Check whether values exist in a foreign column.

  Null values are ignored.

  Args:
    table: Foreign table name.
    column: Foreign column name.

  Example:
    >>> s = pd.Series([0, 1, pd.NA])
    >>> dfs = {'table': pd.DataFrame({'id': [0]})}
    >>> in_foreign_column(s, dfs, table='table', column='id')
    0     True
    1    False
    2     True
    dtype: bool
  """
  # pd.Series.isin false for pd.NA unless pd.NA in reference
  foreign = dfs[table][column]
  return s.isin(foreign) | s.isnull()

@check(message='Not of type {type}')
def is_type(s: pd.Series, *, type: str) -> bool:
  """Check whether column is of a certain data type."""
  return str(s.dtype) == type

# ---- Parsers ----


@check(message='Value could not be parsed to type {type}')
def parse_as_type(s: pd.Series, *, type='string', **kwargs: Any) -> Tuple[pd.Series, pd.Series]:
  """
  Parse column as a certain data type.

  Args:
      type: Data type.

  Raises:
    NotImplementedError: Type not supported.

  Examples:
    >>> valid, result = parse_as_type(pd.Series([0, 1, pd.NA]), type='boolean')
    >>> valid
    0    True
    1    True
    2    True
    dtype: bool
    >>> result
    0    False
    1     True
    2     <NA>
    dtype: boolean
  """
  parser = globals().get(f"parse_{type}")
  if not parser:
    raise NotImplementedError(f"Type {type} not supported")
  # argnames = parser.__code__.co_varnames[1 : parser.__code__.co_argcount]
  # kwargs = {key: getattr(self, key) for key in argnames if hasattr(self, key)}
  parsed = parser(s, **kwargs)
  # pd.testing.assert_index_equal(s.index, parsed.index)
  if parsed is s:
    # TODO: Return scalar True
    return ~pd.Series(dtype=bool, index=s.index), parsed
  return s.isnull() | parsed.notnull(), parsed

def _coerce(s: pd.Series, map: Callable, dtype: str = None) -> pd.Series:
  """Coerce series elements to a specific type, or to null on failure."""
  def fn(x: Any) -> Any:
    try:
      return map(x)
    except (ValueError, TypeError):
      return pd.NA
  s = s.map(fn, na_action='ignore')
  if dtype:
    s = s.astype(dtype, copy=False)
  return s

def parse_string(s: pd.Series) -> pd.Series:
  # pd.api.types.is_string_dtype returns True for object
  if isinstance(s.dtype, pd.StringDtype):
    return s
  return _coerce(s, map=str, dtype='string')

def parse_number(s: pd.Series) -> pd.Series:
  if pd.api.types.is_float_dtype(s):
    return s
  return _coerce(s, map=float, dtype='Float64')

def parse_integer(s: pd.Series) -> pd.Series:
  if pd.api.types.is_integer_dtype(s):
    return s
  return _coerce(s, map=int, dtype='Int64')

def parse_boolean(
  s: pd.Series,
  true_values: List[str] = ["true", "True", "TRUE", "1"],
  false_values: List[str] = ["false", "False", "FALSE", "0"],
) -> pd.Series:
  if pd.api.types.is_bool_dtype(s):
    return s
  if not isinstance(s.dtype, pd.StringDtype):
    s = _coerce(s, map=str)
  new = pd.Series(dtype='boolean', index=s.index)
  new[s.isin(true_values)] = True
  new[s.isin(false_values)] = False
  return new

def parse_date(s: pd.Series, format: str = "default") -> pd.Series:
  if pd.api.types.is_datetime64_dtype(s):
    return s.dt.normalize()
  patterns = {"default": "%Y-%m-%d", "any": None}
  pattern = patterns.get(format, format)
  return pd.to_datetime(
    s, errors='coerce', format=pattern, infer_datetime_format=pattern is None
  ).dt.normalize()

def parse_datetime(s: pd.Series, format: str = 'default') -> pd.Series:
  if pd.api.types.is_datetime64_dtype(s):
    return s
  patterns = {"default": "%Y-%m-%dT%H:%M:%S%z", "any": None}
  pattern = patterns.get(format, format)
  return pd.to_datetime(
    s, errors='coerce', format=pattern, infer_datetime_format=pattern is None
  )

def parse_year(s: pd.Series) -> pd.Series:
  return parse_integer(s)


# ---- Transforms ----

@check
def string_to_lowercase(s: pd.Series) -> Tuple[bool, pd.Series]:
  return True, s.str.lower()

@check
def string_to_uppercase(s: pd.Series) -> Tuple[bool, pd.Series]:
  return True, s.str.upper()

@check
def number_to_integer(s: pd.Series) -> Tuple[bool, pd.Series]:
  return True, s.round().astype('Int64')

@check
def integer_to_number(s: pd.Series) -> Tuple[bool, pd.Series]:
  return True, s.astype('Float64')
