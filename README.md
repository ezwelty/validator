validator
=========

[![codecov](https://codecov.io/gh/ezwelty/validator/branch/main/graph/badge.svg?token=WUANO0HEFG)](https://codecov.io/gh/ezwelty/validator)
[![tests](https://github.com/ezwelty/validator/actions/workflows/tests.yaml/badge.svg)](https://github.com/ezwelty/validator/actions/workflows/tests.yaml)

`validator` makes it easy to validate and transform tabular data in an expressive and reusable way. For example, you can:

- Define a `Schema` against which to validate a `Column`, `Table`, or group of `Tables`, including by converting from a [Frictionless](https://specs.frictionlessdata.io) specification (`convert.frictionless`).
- Write and read a `Schema` to and from text-based metadata (e.g. YAML).
- `Check` the elements of a `Column`, `Table`, or group of `Tables`, optionally in the context of the containing `Table` or `Tables`.
- Explicitly control the order of `Check` execution (e.g. parse strings as integers before checking whether they are greater than 0).
- Conditionally skip a `Check` based on certain criteria.
- Generate a detailed validation `Report`.

*NOTE: This Python package is very much a work in progress. The syntax may still change significantly. Please open an issue if you have any suggestions!*

## Installation

This package is not yet on [PyPI](https://pypi.org), but can still be installed with `pip` directly from this GitHub repository:

```bash
pip install git+https://github.com/ezwelty/validator
```

## Quick start

```py
import pandas as pd
from validator import Schema, Column

# Data to validate
dfs = {
  'main': pd.DataFrame({
    'integer_pk': ['1', '2', '3', pd.NA],
    'number_max': ['1', 'a', '11', pd.NA],
    'boolean': ['TRUE', 'FALSE', 'a', pd.NA]
    }),
  'secondary': pd.DataFrame({
    'integer_pk': ['1', '2', '2', '4'],
    'string_pk': ['a', 'b', 'b', pd.NA]
  })
}

# Data schema
schema = Schema({
  Tables(): Check.only_has_tables(['main', 'secondary']),
  Table('main'): {
    Column('integer_pk'): [
      Check.parse_as_type('integer'),
      Check.not_null(),
      Check.unique()
    ],
    Column('number_max'): [
      Check.parse_as_type('number'),
      Check.less_than_or_equal_to(1),
    ],
    Column('boolean'): Check.parse_as_type('boolean')
  },
  Table('secondary'): {
    Table(): Check.has_columns(['integer_pk', 'string_pk', 'extra'], fill=True),
    Column('integer_pk'): [
      Check.parse_as_type('integer'),
      Check.not_null(),
      Check.in_foreign_column(table='main', column='integer_pk'),
    ],
    Column('string_pk'): [
      Check.parse_as_type('string'),
      Check.not_null()
    ],
    Column('extra'): Check.parse_as_type('boolean'),
    Table(): Check.unique_rows(['integer_pk', 'string_pk'])
  }
})

# Validate the data against the schema
report = schema(dfs)

# Glimpse at the validation report
print(report)
```

```py
Report(Tables(), valid=False, counts={'pass': 8, 'fail': 7})
```

```py
# Dive deeper into the results
df = report.to_dataframe().query('code == "fail"')[
  ['table', 'column', 'row', 'value', 'message']
]
print(df.replace({None: ''}))
```

```
        table      column     row   value                                            message
2        main  integer_pk     [3]  [<NA>]                          Required value is missing
4        main  number_max     [1]     [a]           Value could not be parsed to type number
5        main  number_max     [2]  [11.0]                                          Value > 1
6        main     boolean     [2]     [a]          Value could not be parsed to type boolean
10  secondary  integer_pk     [3]     [4]                       Not found in main.integer_pk
12  secondary   string_pk     [3]  [<NA>]                          Required value is missing
13  secondary              [1, 2]          Duplicate combination of columns ['integer_pk'...
```

```py
# Or access the resulting data, which has been transformed
new_dfs = report.output
print(dfs['secondary'].dtypes, new_dfs['secondary'].dtypes, sep='\n\n')
```

```
integer_pk    object
string_pk     object
dtype: object

integer_pk      Int64
string_pk      string
extra         boolean
dtype: object
```

## Checks

`validator` provides many built-in checks. It is also possible to define and register custom checks.

```py
import yaml
import pandas as pd
from validator import Check, Column, Table, Tables, Schema
from validator.check import register_check

# Define a schema using built-in checks
schema = Schema({
  Tables(): Check.only_has_tables(['main', 'secondary']),
  Table('secondary'): {
    Table(): Check.has_columns(['x']),
    Column('x'): Check.in_foreign_column(
      table='main',
      column='x',
      # Override default message template
      message='Missing from {table}.{column}',
      # Tag for later
      tag='warning'
    )
  }
})

# Register a custom check function
@register_check(
  # Conditionally skip check if df[column] is not present in parent table
  required=lambda column: [Column(column)],
  # Set a default message template
  message='Not equal to {column}'
)
def equal_to_other_column(
  s: pd.Series, df: pd.DataFrame, *, column: str
) -> pd.Series:
  return s.eq(df[column])

# Use it in the schema
schema += Schema({
  Column('y', table='secondary'): Check.equal_to_other_column('x')
})

# Write to text
obj = schema.serialize()
txt = yaml.dump(obj, sort_keys=False)
print(txt)
```

```yml
- checks:
  - name: only_has_tables
    params:
      tables:
      - main
      - secondary
      drop: false
- table: secondary
  schemas:
  - table: null
    checks:
    - name: has_columns
      params:
        columns:
        - x
        fill: false
  - column: x
    checks:
    - name: in_foreign_column
      params:
        table: main
        column: x
      message: Missing from {table}.{column}
      tag: warning
- column: y
  table: secondary
  checks:
  - name: equal_to_other_column
    params:
      column: x
```

```py
# Read back from text
Schema.deserialize(*yaml.safe_load(txt))
```

```py
Schema({
  Tables(): [Check.only_has_tables(tables=['main', 'secondary'], drop=False)],
  Table('secondary'): {
    Table(): [Check.has_columns(columns=['x'], fill=False)],
    Column('x'): [Check.in_foreign_column(table='main', column='x')]
  },
  Column('y', table='secondary'): [Check.equal_to_other_column(column='x')]
})
```

### Check functions explained

The `Check` class accepts a function as its only required parameter. The function for each of the built-in checks can be found under [validator.checks](src/validator/checks).

The function takes either a column (`pd.Series`), table (`pd.DataFrame`) or tables (`Dict[Hashable, DataFrame]`), followed by any needed parent elements (table and/or tables). These parameters must be named `s` / `column`, `df` / `table`, and `dfs` / `tables`, or be defined in `Check.inputs`. Keyword-only parameters (after `*`) may be used to configure the check.

The result may either be a single `Optional[bool]` (`True`: pass, `False`: fail, `None`: skip) or per child (see `Check.axis`) as `Union[Series, Dict[Any, bool]]`. To transform the data, modify it in place (with caution), return the test result and transformation together as a `tuple`, or return the transformed data alone with `Check.test=False`.

## Alternatives

Existing packages that inspired the design of `validator`. Although they did not meet my needs (for the reasons given below), you may find that they meet yours.

- [`pandera`](https://github.com/pandera-dev/pandera)
  - Does not support cross-table checks
  - Does not support transformations (other than type coercion)
- [`frictionless`](https://github.com/frictionlessdata/frictionless-py)
  - Slow for large datasets
  - Does not support cross-table checks (other than foreign keys)
  - Does not support transformations (other than type coercion)
  - Checks that would be trivial to write with `pandas` can be tricky to write for a data stream
- [`great_expectations`](https://github.com/great-expectations/great_expectations)
  - Does not support cross-table checks
  - May not support transformations
- [`goodtables-pandas`](https://github.com/ezwelty/goodtables-pandas-py)
  - Does not support custom checks
  - Does not support transformations (other than type coercion)
  - No longer maintained (by me)
