from typing import Any, Hashable

import pandas as pd
from ..check import check

@check(message='Date does not exist')
def date_exists(s: pd.Series) -> pd.Series:
  # NOTE: Assumes incomplete (99) dates and invalid strings are checked by regex
  valid = pd.Series(dtype='boolean', index=s.index)
  mask = s.notnull()
  year = s[mask].str[0:4].astype(int)
  month = s[mask].str[4:6].astype(int)
  day = s[mask].str[6:8].astype(int)
  is_date = month.ne(99) & day.ne(99)
  parsed = pd.to_datetime(
    pd.DataFrame(
      {'year': year[is_date], 'month': month[is_date], 'day': day[is_date]}
    ),
    errors='coerce'
  )
  valid.loc[is_date.index] = parsed.notnull()
  return valid

@check(message='Winter balance is negative', tag='warning')
def winter_balance_is_positive(s: pd.Series) -> pd.Series:
  return s.ge(0)

@check(message='Summer balance is positive', tag='warning')
def summer_balance_is_negative(s: pd.Series) -> pd.Series:
  return s.le(0)

@check(message='Value is null although {column} is {value}')
def not_null_if_column_equal_to(s: pd.Series, df: pd.DataFrame, *, column: Hashable, value: Any) -> pd.Series:
  return df[column].ne(value) | (df[column].eq(value) & s.notnull())

@check(message='Value is not null although {column} is null')
def null_if_column_null(s: pd.Series, df: pd.DataFrame, *, column: Hashable) -> pd.Series:
  return df[column].notnull() | (df[column].isnull() & s.isnull())

@check(message='Value is greater than {column}')
def less_than_or_equal_to_column(s: pd.Series, df: pd.DataFrame, *, column: Hashable) -> pd.Series:
  return s.le(df[column])

@check(message='Not within {round(100 * tolerance)}% of thickness change estimated from VOLUME_CHANGE, AREA, and AREA_CHANGE', tag='warning')
def consistent_with_estimated_thickness_change(s: pd.DataFrame, df: pd.DataFrame, *, tolerance: float = 0.0) -> pd.Series:
  actual = s / 1e3
  predicted = df['VOLUME_CHANGE'] * 1e3 / (df['AREA'] * 1e6 - 0.5 * df['AREA_CHANGE'] * 1e3)
  return abs(actual - predicted) / actual < abs(tolerance)

@check(message='Glacier-wide ANNUAL_BALANCE not within {round(100 * tolerance)}% of estimate from bands ANNUAL_BALANCE and AREA', tag='warning')
def glacier_and_band_balance_are_consistent(df: pd.DataFrame, *, tolerance: float = 0.0) -> pd.Series:

  # TODO: Custom error messages based on failure type?
  # TODO: Move groupby to check execution?
  def _fn(gdf: pd.DataFrame) -> bool:
    is_band = gdf['LOWER_BOUND'].notnull() & gdf['UPPER_BOUND'].notnull()
    is_glacier = gdf['LOWER_BOUND'].isnull() & gdf['UPPER_BOUND'].isnull()
    if is_band.sum().eq(0) or is_glacier.sum().eq(0):
      return pd.NA
    if is_glacier.sum().gt(1):
      return False
    actual = gdf[is_glacier]['ANNUAL_BALANCE'].iloc[0]
    predicted = (gdf[is_band]['ANNUAL_BALANCE'] * gdf[is_band]['AREA']).sum() / gdf[is_glacier]['AREA'].iloc[0]
    return abs(actual - predicted) / actual < abs(tolerance)

  groupby = df.groupby(['WGMS_ID', 'YEAR'])
  return groupby.transform(_fn).astype('boolean')
