import pandas as pd

from validator import Check, Column, Schema, Table, Tables

# ---- Data ----

dfs = {
  'main': pd.DataFrame({
    'integer_pk': ['1', '2', '3', ''],
    'integer_minmax': ['1', '1.2', '11', ''],
    'number_minmax': ['1', 'a', '11', ''],
    'boolean': ['TRUE', 'FALSE', 'a', ''],
    'string_minmax': ['ab', 'a', 'abcd', ''],
    'string_enum': ['a', 'b', 'f', ''],
    'string_regex': ['a', 'b', 'f', ''],
    }).replace('', pd.NA),
  'secondary': pd.DataFrame({
    'integer_pk': ['1', '2', '2', '4'],
    'string_pk': ['a', 'b', 'b', ''],
    'boolean_lookup': ['TRUE', 'TRUE', 'FALSE', '']
  }).replace('', pd.NA)
}

# ---- Schema ----

schema = Schema({
  Tables(): [
    Check.only_has_tables(['main', 'secondary'])
  ],
  Table('main'): {
    Column('integer_pk'): [
      Check.parse_as_type('integer'),
      Check.not_null(),
      Check.unique()
    ],
    Column('integer_minmax'): [
      Check.parse_as_type('integer'),
      Check.greater_than_or_equal_to(1),
      Check.less_than_or_equal_to(10)
    ],
    Column('number_minmax'): [
      Check.parse_as_type('number'),
      Check.greater_than_or_equal_to(1),
      Check.less_than_or_equal_to(1),
    ],
    Column('boolean'): [
      Check.parse_as_type('boolean')
    ],
    Column('string_minmax'): [
      Check.length_greater_than_or_equal_to(2),
      Check.length_less_than_or_equal_to(3)
    ],
    Column('string_enum'): [
      Check.in_list(['a', 'b'])
    ],
    Column('string_regex'): [
      Check.matches_regex(r'[ab]')
    ]
  },
  Table('secondary'): {
    Table(): [
      Check.has_columns(['integer_pk', 'string_pk', 'boolean_lookup', 'extra'], coerce=True),
      Check.only_has_columns(['integer_pk', 'string_pk', 'boolean_lookup'])
    ],
    Column('integer_pk'): [
      Check.parse_as_type('integer'),
      Check.not_null(),
      Check.in_foreign_column(table='main', column='integer_pk'),
      Check.in_column(column='integer_pk_missing'),
      Check.in_foreign_column(table='main', column='integer_pk_missing')
    ],
    Column('string_pk'): [
      Check.not_null()
    ],
    Column('boolean_lookup'): [
      Check.parse_as_type('boolean')
    ],
    Table(): [
      Check.unique_rows(['integer_pk', 'string_pk']),
      Check.matches_foreign_columns(table='main', join={'integer_pk': 'integer_pk'}, columns={'boolean_lookup': 'boolean'})
    ]
  },
})

# ---- Expected results ----

failures = pd.DataFrame([
  {
    'table': 'main',
    'column': 'integer_pk',
    'check': Check.not_null(),
    'row': [3],
    'value': [pd.NA]
  },
  {
    'table': 'main',
    'column': 'integer_minmax',
    'check': Check.parse_as_type('integer'),
    'row': [1],
    'value': ['1.2']
  },
  {
    'table': 'main',
    'column': 'integer_minmax',
    'check': Check.less_than_or_equal_to(10),
    'row': [2],
    'value': [11]
  },
  {
    'table': 'main',
    'column': 'number_minmax',
    'check': Check.parse_as_type('number'),
    'row': [1],
    'value': ['a']
  },
  {
    'table': 'main',
    'column': 'number_minmax',
    'check': Check.less_than_or_equal_to(1),
    'row': [2],
    'value': [11.0]
  },
  {
    'table': 'main',
    'column': 'boolean',
    'check': Check.parse_as_type('boolean'),
    'row': [2],
    'value': ['a']
  },
  {
    'table': 'main',
    'column': 'string_minmax',
    'check': Check.length_greater_than_or_equal_to(2),
    'row': [1],
    'value': ['a']
  },
  {
    'table': 'main',
    'column': 'string_minmax',
    'check': Check.length_less_than_or_equal_to(3),
    'row': [2],
    'value': ['abcd']
  },
  {
    'table': 'main',
    'column': 'string_enum',
    'check': Check.in_list(['a', 'b']),
    'row': [2],
    'value': ['f']
  },
  {
    'table': 'main',
    'column': 'string_regex',
    'check': Check.matches_regex(r'[ab]'),
    'row': [2],
    'value': ['f']
  },
  {
    'table': 'secondary',
    'column': ['extra'],
    'check': Check.only_has_columns(['integer_pk', 'string_pk', 'boolean_lookup'])
  },
  {
    'table': 'secondary',
    'column': 'integer_pk',
    'check': Check.in_foreign_column(table='main', column='integer_pk'),
    'row': [3],
    'value': [4]
  },
  {
    'table': 'secondary',
    'column': 'string_pk',
    'check': Check.not_null(),
    'row': [3],
    'value': [pd.NA]
  },
  {
    'table': 'secondary',
    'check': Check.unique_rows(columns=['integer_pk', 'string_pk']),
    'row': [1, 2]
  },
  {
    'table': 'secondary',
    'check': Check.matches_foreign_columns(table='main', join={'integer_pk': 'integer_pk'}, columns={'boolean_lookup': 'boolean'}),
    'row': [1]
  }
])

skips = pd.DataFrame([
  {
    'table': 'secondary',
    'column': 'integer_pk',
    'check': Check.in_column(column='integer_pk_missing'),
    'message': "Missing required inputs [Column('integer_pk_missing')]"
  },
  {
    'table': 'secondary',
    'column': 'integer_pk',
    'check': Check.in_foreign_column(table='main', column='integer_pk_missing'),
    'message': "Missing required inputs [Column('integer_pk_missing', table='main')]"
  }
])

# ---- Tests ----

def test_checks_child_column_without_required_parents() -> None:
  table = 'main'
  column = 'integer_pk'
  report = schema(dfs, target=Column(column, table))
  assert report.target.equals(Column(column, table))
  assert report.counts == {'pass': 2, 'fail': 1}
  reports = [
    schema(dfs[table], name=Table(table), target=Column(column, table)),
    schema(dfs[table][column], name=Column(column, table)),
    schema(dfs[table][column], name=Column(column, table), target=Column(column, table))
  ]
  assert all(r.target.equals(report.target) for r in reports)
  assert all(r.counts == report.counts for r in reports)

def test_checks_child_column_with_required_parents() -> None:
  table = 'secondary'
  column = 'integer_pk'
  report = schema(dfs, target=Column(column, table))
  assert report.target.equals(Column(column, table))
  assert report.counts == {'pass': 2, 'fail': 1, 'skip': 2}
  reports = [
    schema(dfs[table], name=Table(table), target=Column(column, table)),
    schema(dfs[table][column], name=Column(column, table)),
    schema(dfs[table][column], name=Column(column, table), target=Column(column, table))
  ]
  assert all(r.target.equals(report.target) for r in reports)
  assert all(r.counts == {'pass': 2, 'skip': 3} for r in reports)

def test_checks_child_table_without_required_parents() -> None:
  table = 'main'
  report = schema(dfs, target=Table(table))
  assert report.target.equals(Table(table))
  assert report.counts == {'pass': 4, 'fail': 10}
  reports = [
    schema(dfs[table], name=Table(table), target=Table(table)),
    schema(dfs[table], name=Table(table))
  ]
  assert all(r.target.equals(report.target) for r in reports)
  assert all(r.counts == report.counts for r in reports)

def test_checks_child_table_with_required_parents() -> None:
  table = 'secondary'
  report = schema(dfs, target=Table(table))
  assert report.target.equals(Table(table))
  # Error caused by foreign key check on non-parsed reference
  assert report.counts == {'pass': 4, 'fail': 4, 'skip': 2, 'error': 1}
  reports = [
    schema(dfs[table], name=Table(table), target=Table(table)),
    schema(dfs[table], name=Table(table))
  ]
  assert all(r.target.equals(report.target) for r in reports)
  assert all(r.counts == {'pass': 4, 'fail': 3, 'skip': 4} for r in reports)

def test_checks_tables() -> None:
  report = schema(dfs)
  assert report.target.equals(Tables())
  assert report.valid is False
  assert report.counts == {'pass': 9, 'fail': 15, 'skip': 2}
  reports = [
    schema(dfs, name=Tables()),
    schema(dfs, name=Tables(), target=Tables())
  ]
  assert all(r.target.equals(report.target) for r in reports)
  assert all(r.counts == report.counts for r in reports)
  df = report.to_dataframe()
  pd.testing.assert_frame_equal(
    failures.where(failures.notnull(), None),
    df[df['code'].eq('fail')][failures.columns].reset_index(drop=True)
  )
  pd.testing.assert_frame_equal(
    skips.where(skips.notnull(), None),
    df[df['code'].eq('skip')][skips.columns].reset_index(drop=True)
  )
