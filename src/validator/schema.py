import copy as copylib
from typing import Dict, List, Optional

import pandas as pd

from .check import Check, Result
from .helpers import stringify_call, filter_kwargs
from .targets import Target, Column, Table, Tables, classify_data, extract_data
from .types import Data, FlatSchemaDict, SchemaDict

# ---- Schema ----

class Schema:
  """
  Tests and transforms tabular data following an ordered sequence of checks.

  Parameters
  ----------
  schema
    Schema definition as a dictionary, where each key is a :class:`Target`
    and each value is either a :type:`list` of :class:`Check` or a dictionary
    of subschemas. Since each :class:`Target` instance is unique,
    they can repeated at will (e.g. `{Column(): [], Column(): []}`).

    Any nesting is allowed, with two exceptions:
    - The target column or table name cannot be changed once set. For example,
    `{Table('X'): {Table('X'): []}}` and `{Table('X'): {Table(): []}}`
    are valid (`Table()` inherits the table name from `Table('x')`), but
    `{Table('X'): {Table('Y'): []}}` is not.
    - Unnamed columns cannot be nested under named tables. For example,
    `{Table('X'): {Column('Y'): []}}` is valid but
    `{Table('X'): {Column(): []}}` is not.

  Raises
  ------
  ValueError
    Invalid schema (with a detailed list of errors).
  """
  def __init__(self, schema: SchemaDict) -> None:
    if not isinstance(schema, dict):
      raise TypeError(f'Schema must be a dictionary, not a {type(schema)}')
    self.schema = schema
    self._flatten()

  def __repr__(self) -> str:
    return stringify_call(self.__class__.__name__, self.schema)

  def _flatten(self) -> FlatSchemaDict:
    """
    Flatten schema.

    Examples
    --------
    >>> schema = Schema({
    ...   Table('X'): {
    ...     Table(): [Check(lambda df: 'x' in df, name='has_x')],
    ...     Column('x'): [Check(lambda s: s.notnull(), name='x_not_null')]
    ...   }
    ... })
    >>> schema._flatten()
    {Table('X'): Check.has_x(), Column('x', table='X'): Check.x_not_null()}
    """
    errors = []
    flat = {}

    def traverse(
      schema: SchemaDict,
      baseprefix: str = '',
      basenames: dict = {}
    ) -> None:
      for key, value in schema.items():
        prefix = f'{baseprefix}.{key}'
        # Check class of dictionary key
        if not isinstance(key, Target):
          errors.append(f'{prefix}: Not a target (Column, Table, or Tables)')
          continue
        # Check that table and column names are not changed if already set
        key_names = {k: v for k, v in key.__dict__.items() if v is not None}
        name_changes = {
          basenames[k]: v for k, v in key_names.items()
          if basenames.get(k) is not None and v != basenames[k]
        }
        if name_changes:
          errors.append(f'{prefix}: Changes target names {name_changes}')
          continue
        names = {**basenames, **key_names}
        # Check valid names
        try:
          flat_key = filter_kwargs(type(key), **names)
        except ValueError as error:
          errors.append(f'{prefix}: {error}')
          continue
        # Parse dictionary value
        if isinstance(value, list):
          for i, check in enumerate(value):
            if isinstance(check, Check):
              cls = list(check.inputs.values())[0]
              if cls is not type(key):
                errors.append(
                  f'{prefix}[{i}]: Check has wrong target class ({cls})'
                )
              else:
                flat[flat_key.copy()] = check
            else:
              errors.append(f'{prefix}[{i}]: Expected Check, not {type(check)}')
        elif isinstance(value, dict):
          traverse(value, baseprefix=prefix, basenames=names)
        else:
          errors.append(f'{prefix}: Expected list or dict, not {type(value)}')

    traverse(self.schema)
    if errors:
      raise ValueError('Invalid schema.\n\n' + '\n'.join(errors))
    return flat

  def _squeeze(self, n: int = 1) -> 'Schema':
    """
    Squeeze redundancy out of a schema.

    Examples
    --------
    >>> schema = Schema({
    ...   Table('x'): {
    ...     Table(): [],
    ...     Table(): [Check.has_columns(['x'])]
    ...   },
    ...   Table('x'): {
    ...     Column('x'): [Check.not_null()],
    ...     Column('x'): [Check.unique()]
    ...   }
    ... })
    >>> schema._squeeze()
    Schema({Table('x'):
        [Check.has_columns(columns=['x'], fill=False)],
      Table('x'):
        {Column('x'):
          [Check.not_null(), Check.unique()]}})
    """
    # Drop keys with no checks
    def drop_empty(obj: SchemaDict) -> None:
      drop = []
      for key, value in obj.items():
        if not value:
          drop.append(key)
        elif isinstance(value, dict):
          drop_empty(value)
      for key in drop:
        del obj[key]
    # Merge checks of equal sequential keys
    def merge_equal_siblings(obj: SchemaDict) -> SchemaDict:
      base = None
      drop = []
      for key, value in obj.items():
        if isinstance(value, dict):
          obj[key] = merge_equal_siblings(value)
        else:
          if base is None or not base.equals(key):
            base = key
          elif base and not isinstance(value, dict):
            obj[base] += value
            drop.append(key)
      for key in drop:
        del obj[key]
      return obj
    # Unwrap subschema if single key and same class as key
    def unwrap_singletons(obj: SchemaDict, base: Target = None) -> SchemaDict:
      if len(obj) == 1:
        key = list(obj.keys())[0]
        if type(key) is type(base):
          value = obj[key]
          if isinstance(value, dict):
            return unwrap_singletons(value, base=base)
          return value
      for key, value in obj.items():
        if isinstance(value, dict):
          obj[key] = unwrap_singletons(value, base=key)
      return obj
    for _ in range(n):
      drop_empty(self.schema)
      merge_equal_siblings(self.schema)
      unwrap_singletons(self.schema)
    return self

  def _filter(self, input: Data, target: Target) -> FlatSchemaDict:
    """
    Filter schema after flattening based on input data and name.

    Examples
    --------
    >>> schema = Schema({
    ...   Table(): [Check.has_columns(['x'])],
    ...   Column(): [Check.not_null()],
    ...   Column('x'): [Check.unique()],
    ... })
    >>> dfs = {'a': pd.DataFrame({'x': [0], 'y': [1]})}
    >>> schema._filter(dfs, Tables())
    {Table('a'): Check.has_columns(columns=['x'], fill=False),
    Column('x', table='a'): Check.not_null(),
    Column('y', table='a'): Check.not_null(),
    Column('x', table='a'): Check.unique()}
    >>> schema._filter(dfs['a'], Table('a'))
    {Table('a'): Check.has_columns(columns=['x'], fill=False),
    Column('x', table='a'): Check.not_null(),
    Column('y', table='a'): Check.not_null(),
    Column('x', table='a'): Check.unique()}
    >>> schema._filter(dfs['a'], Table())
    {Table(): Check.has_columns(columns=['x'], fill=False),
    Column('x'): Check.not_null(),
    Column('y'): Check.not_null(),
    Column('x'): Check.unique()}
    >>> schema._filter(dfs['a']['x'], Column('x', table='a'))
    {Column('x', table='a'): Check.not_null(),
    Column('x', table='a'): Check.unique()}
    >>> schema._filter(dfs['a']['x'], Column('x'))
    {Column('x'): Check.not_null(), Column('x'): Check.unique()}
    >>> schema._filter(dfs['a']['x'], Column())
    {Column(): Check.not_null()}
    """
    filtered = {}
    for key, check in self._flatten().items():
      if key.equals(target):
        filtered[key] = check
      elif key.matches(target):
        filtered[target.copy()] = check
      elif key in target:
        if isinstance(key, (Column, Table)) and key.table is not None:
          filtered[key] = check
        elif isinstance(key, Table):
          filtered.update({Table(table): check for table in input})
        elif isinstance(key, Column):
          if isinstance(target, Table):
            columns = list(input) if key.column is None else [key.column]
            filtered.update(
              {Column(column, target.table): check for column in columns}
            )
          elif isinstance(target, Tables):
            for table in input:
              filtered.update(
                {
                  Column(column, table): check
                  for column in
                  (input[table] if key.column is None else [key.column])
                  if column in input[table]
                }
              )
    return filtered

  def __call__(
    self,
    data: Data = None,
    name: Target = None,
    target: Target = None,
    copy: bool = True
  ) -> 'Report':
    """
    Run the checks on the provided tabular data.

    Executes each applicable check and wraps the results in a :class:`Report`.

    Parameters
    ----------
    data
      Tabular data.
    name
      Class and name of `data` (e.g. `Column('x')`).
      If not provided, attempts to guess the class based on the type of `data`
      (e.g. `Table()` for :class:`pandas.DataFrame`).
    target
      Name of the tabular element to check. Defaults to `name`.
      If provided, must be a child of `name` (e.g. `Column('x')` in `Table()`).
    copy
      Whether to process a (deep) copy of `data`.
      If any checks transform their input,
      this ensures that the changes do not propagate to the original `data`.

    Examples
    --------
    >>> schema = Schema({
    ...   Column(): [Check.not_null()],
    ...   Column('x'): [Check.unique()]
    ... })
    >>> s = pd.Series([0, 1])
    >>> schema(s).counts
    {'pass': 1}
    >>> schema(s, name=Column('x')).counts
    {'pass': 2}

    >>> df = pd.DataFrame({'x': s})
    >>> schema(df, target=Column('x')).counts
    {'pass': 2}

    >>> df = pd.DataFrame({'y': s})
    >>> schema(df).results[1].message
    "Missing required inputs [Column('x')]"
    """
    if copy:
      data = copylib.deepcopy(data)
    inputs = extract_data(data, name=name, target=target)
    name = name or classify_data(data)()
    target = target or name
    input = inputs[type(target)]
    results = {}
    for key, check in self._filter(input, target).items():
      # print(f'{key}: {check}')
      # Load data for check
      args = inputs.copy()
      if isinstance(target, Tables) and isinstance(key, (Table, Column)):
        if not key.table in args[Tables]:
          # Table {key.table} not in tables
          results[key] = Result(check, target=key, missing=[Table(key.table)])
          continue
        args[Table] = args[Tables][key.table]
      if isinstance(target, (Tables, Table)) and isinstance(key, Column):
        if not key.column in args[Table]:
          # Column {key.column} not in table {key.table}
          results[key] = Result(
            check, target=key, missing=[Column(key.column, table=key.table)]
          )
          continue
        args[Column] = args[Table][key.column]
      # Run check
      check_class = list(check.inputs.values())[0]
      result = check(data, name=name, target=key)
      # Reassign new value
      output = result.output
      if output is not None and output is not args[check_class]:
        if type(target) is check_class:
          inputs[check_class] = output
        elif type(target) is Tables:
          if check_class is Table:
            inputs[Tables][key.table] = output
          else:
            # Column
            inputs[Tables][key.table][key.column] = output
        elif type(target) is Table:
          # Column
          inputs[Table][key.column] = output
      results[key] = result
    output = inputs[type(target)]
    return Report(
      list(results.values()), target=target, input=input, output=output
    )

  def serialize(self) -> List[dict]:
    """
    Get the dictionary representation of a schema.

    Only registered checks (created using a `Check` constructor) are supported.
    See :meth:`Check.serialize` for details.

    Examples
    --------
    >>> import yaml
    >>> from validator.check import register_check
    >>>
    >>> @register_check
    ... def has_column(df, *, column):
    ...   return column in df
    >>>
    >>> @register_check
    ... def in_values(s, *, values):
    ...   return s.isin(values)
    >>>
    >>> schema = Schema({Table('X'): {
    ...   Table(): [Check.has_column('x')],
    ...   Column('x'): [Check.in_values([0, 1])]
    ... }})
    >>> d = schema.serialize()
    >>> print(yaml.dump(d, sort_keys=False))
    - table: X
      schemas:
      - table: null
        checks:
        - name: has_column
          params:
            column: x
      - column: x
        checks:
        - name: in_values
          params:
            values:
            - 0
            - 1
    >>> Schema.deserialize(*d).serialize() == d
    True
    """
    def _serialize(schema: SchemaDict) -> dict:
      items = []
      for target, subvalue in schema.items():
        item = {**target.__dict__}
        if isinstance(target, Column) and item['table'] is None:
          del item['table']
        if isinstance(subvalue, list):
          item['checks'] = [check.serialize() for check in subvalue]
        else:
          item['schemas'] = _serialize(subvalue)
        items.append(item)
      return items

    return _serialize(self.schema)

  @classmethod
  def deserialize(cls, *schemas: dict) -> 'Schema':
    """
    Load a schema from dictionaries.

    Only registered checks (created using a `Check` constructor) are supported.
    See :meth:`Check.deserialize` for details.

    Parameters
    ----------
    *schemas
      Dictionaries representing a schema.
    """
    def _deserialize(schemas: List[dict]) -> dict:
      obj = {}
      for schema in schemas:
        temp = {k: schema[k] for k in ['table', 'column'] if k in schema}
        target = Target.create(**temp)
        if 'checks' in schema and 'schemas' in schema:
          raise ValueError("Schema cannot have both 'schemas' and 'checks'")
        if 'checks' in schema:
          obj[target] = [
            Check.deserialize(**check) for check in schema['checks']
          ]
        elif 'schemas' in schema:
          obj[target] = _deserialize(schema['schemas'])
        else:
          raise ValueError("Schema must have either 'schemas' or 'checks'")
      return obj

    return cls(_deserialize(schemas))

  def __add__(self, other: 'Schema') -> 'Schema':
    """
    Concatenate two schemas.

    Parameters
    ----------
    other
      Schema to concatenate to the end of the first schema.

    Examples
    --------
    >>> x = Schema({Column('x'): [Check(lambda s: s.notnull())]})
    >>> y = Schema({Column('y'): [Check(lambda s: s.notnull())]})
    >>> (x + y).schema
    {Column('x'): [Check.<lambda>()], Column('y'): [Check.<lambda>()]}
    """
    if not isinstance(other, self.__class__):
      return other + self
    return self.__class__({**self.schema, **other.schema})


class Report:
  """
  Describes the result of a :class:`Schema` run.

  Parameters
  ----------
  results
    Result of each `Check`.
  target
    Target of the run.
  input
    Input data for `target`.
  output
    Output (potentially transformed) data for `target`.
  """

  def __init__(
    self,
    results: List[Result],
    target: Target,
    input: Data = None,
    output: Data = None
  ) -> None:
    self.results = results
    self.target = target
    self.input = input
    self.output = output

  def __repr__(self) -> str:
    return stringify_call(
      self.__class__.__name__, self.target, valid=self.valid, counts=self.counts
    )

  @property
  def counts(self) -> Dict[str, int]:
    """Number of results by result code."""
    n = {}
    for result in self.results:
      if result.code in n:
        n[result.code] += 1
      else:
        n[result.code] = 1
    return n

  @property
  def valid(self) -> Optional[bool]:
    """
    Overall test result.

    - `None` if :attr:`results` is empty or all result codes are 'skip'
    - `True` if all result codes are 'pass' or 'skip'
    - `False` otherwise
    """
    if not self.results or all(result.code == 'skip' for result in self.results):
      return None
    return all(result.code in ['pass', 'skip'] for result in self.results)

  def to_dataframe(self, explode: bool = False) -> pd.DataFrame:
    """
    Represent report as a :class:`pandas.DataFrame`.

    Parameters
    ----------
    explode
      Whether to expand `Result` with a non-scalar test result to multiple rows.

    Examples
    --------
    >>> schema = Schema({Column(): [Check(lambda s: s.gt(2), name='gt2')]})
    >>> report = schema(pd.Series([1, 2, 3]))
    >>> report.to_dataframe()
       code table column     row   value        check   tag message
    0  fail  None   None  [0, 1]  [1, 2]  Check.gt2()  None    None
    >>> report.to_dataframe(explode=True)
       code table column row value        check   tag message
    0  fail  None   None   0     1  Check.gt2()  None    None
    0  fail  None   None   1     2  Check.gt2()  None    None
    """
    dicts = [result._to_dict() for result in self.results]
    df = pd.DataFrame(dicts)
    if explode:
      df = df.explode('table').explode('column')
      mask = df['value'].isnull() & df['row'].notnull()
      df = pd.concat([
        df[~mask].explode(['row', 'value']),
        df[mask].explode('row')
      ])
      df.sort_index(inplace=True)
    return df
