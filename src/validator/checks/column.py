"""Column-level check functions."""
import datetime
import re
from typing import Any, Callable, Dict, Hashable, Iterable, List, Tuple, Union

import pandas as pd

from ..check import register_check
from ..targets import Column

# ---- Checks ----


@register_check(message='Required value is missing')
def not_null(s: pd.Series) -> pd.Series:
    """Check whether values are not null."""
    return s.notnull()


@register_check(message='Duplicate value')
def unique(s: pd.Series) -> pd.Series:
    """
    Check whether values are unique.

    Null values are ignored.

    Examples
    --------
    >>> s = pd.Series([0, 1, 1, pd.NA, pd.NA])
    >>> unique(s)
    0     True
    1    False
    2    False
    dtype: bool
    """
    return ~s.dropna().duplicated(keep=False)


@register_check(message='Value not in {values}')
def in_list(s: pd.Series, *, values: Iterable) -> pd.Series:
    """
    Check whether values are in a set of allowed values.

    Examples
    --------
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


@register_check(message='Value < {min}')
def greater_than_or_equal_to(s: pd.Series, *, min: Any) -> pd.Series:
    """Check whether values are greater than or equal to a minimum."""
    if pd.api.types.is_extension_array_dtype(s):
        return s.ge(min)
    return s.isnull() | s.ge(min)


@register_check(message='Value > {max}')
def less_than_or_equal_to(s: pd.Series, *, max: Any) -> pd.Series:
    """Check whether values are less than or equal to a maximum."""
    if pd.api.types.is_extension_array_dtype(s):
        return s.le(max)
    return s.isnull() | s.le(max)


@register_check(message='Length < {min}')
def length_greater_than_or_equal_to(s: pd.Series, *, min: int) -> pd.Series:
    """Check whether value lengths are greater than or equal to a minimum."""
    # Supports sequences (string, tuple, list) and collections (dictionary)
    # Conversion to Int64 needed from object input to ensure nulls are ignored
    return s.str.len().astype('Int64', copy=False).ge(min)


@register_check(message='Length > {max}')
def length_less_than_or_equal_to(s: pd.Series, *, max: int) -> pd.Series:
    """Check whether value lengths are less than or equal to a maximum."""
    # Supports sequences (string, tuple, list) and collections (dictionary)
    # Conversion to Int64 needed from object input to ensure nulls are ignored
    return s.str.len().astype('Int64', copy=False).le(max)


@register_check(message='Value does not match regular expression {regex}')
def matches_regex(s: pd.Series, *, regex: str) -> pd.Series:
    """Check whether values match a regular expression."""
    return s.str.fullmatch(regex).astype('boolean')


@register_check(
    message='Not found in {column}', required=lambda column: [Column(column)]
)
def in_column(s: pd.Series, df: pd.DataFrame, *, column: Hashable) -> pd.Series:
    """Check whether values exist in another column."""
    other = df[column]
    return s.isin(other) | s.isnull()


@register_check(
    message='Not found in {table}.{column}',
    required=lambda table, column: [Column(column, table=table)],
)
def in_foreign_column(
    s: pd.Series,
    dfs: Dict[Hashable, pd.DataFrame],
    *,
    table: Hashable,
    column: Hashable,
) -> pd.Series:
    """
    Check whether values exist in a foreign column.

    Null values are ignored.

    Parameters
    ----------
    table
        Foreign table name.
    column
        Foreign column name.

    Examples
    --------
    >>> s = pd.Series([0, 1, pd.NA])
    >>> dfs = {'table': pd.DataFrame({'id': [0]})}
    >>> in_foreign_column(s, dfs, table='table', column='id')
    0     True
    1    False
    dtype: bool
    """
    foreign = dfs[table][column]
    return s[not_null].isin(foreign)


@register_check(message='Not of type {type}')
def is_type(s: pd.Series, *, type: str) -> bool:
    """Check whether column is of a certain data type."""
    return str(s.dtype) == type


# ---- Parsers ----


@register_check(message='Value could not be parsed to type {type}')
def parse_as_type(
    s: pd.Series, *, type: str = 'string', **kwargs: Any
) -> Tuple[Union[bool, pd.Series], pd.Series]:
    """
    Parse column as a certain data type.

    Parameters
    ----------
    type
        Data type.

    Raises
    ------
    NotImplementedError
        Type not supported.

    Examples
    --------
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
    parser = globals().get(f'parse_{type}')
    if not parser:
        raise NotImplementedError(f'Type {type} not supported')
    # argnames = parser.__code__.co_varnames[1 : parser.__code__.co_argcount]
    # kwargs = {key: getattr(self, key) for key in argnames if hasattr(self, key)}
    parsed = parser(s, **kwargs)
    # pd.testing.assert_index_equal(s.index, parsed.index)
    if parsed is s:
        return True, parsed
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
    """Parse values as strings."""
    # pd.api.types.is_string_dtype returns True for object
    if isinstance(s.dtype, pd.StringDtype):
        return s
    return _coerce(s, map=str, dtype='string')


def parse_number(s: pd.Series) -> pd.Series:
    """Parse values as numbers."""
    if pd.api.types.is_float_dtype(s):
        return s
    return _coerce(s, map=float, dtype='Float64')


def parse_integer(s: pd.Series) -> pd.Series:
    """Parse values as integers."""
    if pd.api.types.is_integer_dtype(s):
        return s
    return _coerce(s, map=int, dtype='Int64')


def parse_boolean(
    s: pd.Series,
    true_values: List[str] = ['true', 'True', 'TRUE', '1'],
    false_values: List[str] = ['false', 'False', 'FALSE', '0'],
) -> pd.Series:
    """Parse values as booleans."""
    if pd.api.types.is_bool_dtype(s):
        return s
    if not isinstance(s.dtype, pd.StringDtype):
        s = _coerce(s, map=str)
    new = pd.Series(dtype='boolean', index=s.index)
    new[s.isin(true_values)] = True
    new[s.isin(false_values)] = False
    return new


def _is_date(s: pd.Series) -> bool:
    """Check whether a series is of date(time) type."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    values = s.dropna()
    if values.empty:
        return True
    if isinstance(values.iloc[0], (datetime.date, datetime.datetime)):
        return True
    return False


def _is_datetime(s: pd.Series) -> bool:
    """Check whether a series is of datetime type."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    values = s.dropna()
    if values.empty:
        return True
    if isinstance(values.iloc[0], datetime.datetime):
        return True
    return False


def parse_date(s: pd.Series, format: str = 'default') -> pd.Series:
    """Parse values (strings only) as dates."""
    if _is_date(s):
        return s
    if format == 'any':
        raise NotImplementedError("Date format 'any' not supported")
    if format == 'default' or format == '%Y-%m-%d':
        return _coerce(s, map=datetime.date.fromisoformat)
    return _coerce(s, map=lambda x: datetime.datetime.strptime(x, format).date())


def parse_datetime(s: pd.Series, format: str = 'default') -> pd.Series:
    """Parse values (strings only) as datetimes."""
    if _is_datetime(s):
        return s
    if format == 'any':
        raise NotImplementedError("Datetime format 'any' not supported")
    if format == 'default' or re.fullmatch(
        r'%Y-%m-%d([T ]%H(:%M(:%S(\.%f)?)?)?(%z)?)?', format
    ):
        return _coerce(s, map=datetime.datetime.fromisoformat)
    return _coerce(s, map=lambda x: datetime.datetime.strptime(x, format))


def parse_year(s: pd.Series) -> pd.Series:
    """Parse values as years."""
    return parse_integer(s)


# ---- Transforms ----


@register_check
def string_to_lowercase(s: pd.Series) -> pd.Series:
    """Convert strings to lowercase."""
    return s.str.lower()


@register_check
def string_to_uppercase(s: pd.Series) -> pd.Series:
    """Convert strings to uppercase."""
    return s.str.upper()


@register_check
def number_to_integer(s: pd.Series) -> pd.Series:
    """Round numbers and cast them to integer."""
    return s.round().astype('Int64')


@register_check
def integer_to_number(s: pd.Series) -> pd.Series:
    """Cast integers to number."""
    return s.astype('Float64')
