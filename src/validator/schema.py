import copy as copylib
from typing import Dict, List, Optional, Tuple

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
    and each value is either a :class:`Check`, :type:`list` of :class:`Check`,
    or a dictionary of subschemas. Since each :class:`Target` is unique,
    they can repeated at will (e.g. `{Column(): [], Column(): []}`).

    Any nesting is allowed, except that the target column or table name cannot
    be changed once set. For example, `{Table('X'): {Table('X'): []}}` and
    `{Table('X'): {Table(): []}}` are valid (`Table()` inherits the table name
    from `Table('x')`), but `{Table('X'): {Table('Y'): []}}` is not.

    Wildcard targets are applied to all elements available at check runtime.
    For example, `Column()` is applied to all columns in the data by the time
    it is evaluated.

  Raises
  ------
  ValueError
    Invalid schema (with a detailed list of errors).
  """
  def __init__(self, schema: SchemaDict) -> None:
    if not isinstance(schema, dict):
      raise TypeError(f'Schema must be a dictionary, not a {type(schema)}')
    type(self)._flatten(schema)
    self.schema = schema

  def __repr__(self) -> str:
    return stringify_call(self.__class__.__name__, self.schema)

  @staticmethod
  def _flatten(checks: SchemaDict) -> FlatSchemaDict:
    """
    Flatten checks.

    Examples
    --------
    >>> checks = {
    ...   Table('X'): {
    ...     Table(): [Check(lambda df: 'x' in df, name='has_x')],
    ...     Column('x'): [Check(lambda s: s.notnull(), name='x_not_null')]
    ...   }
    ... }
    >>> Schema._flatten(checks)
    {Table('X'): Check.has_x(), Column('x', table='X'): Check.x_not_null()}
    """
    errors = []
    flat = {}

    def traverse(
      checks: SchemaDict,
      baseprefix: str = '',
      basenames: dict = {}
    ) -> None:
      for key, value in checks.items():
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
        if isinstance(value, Check):
          value = [value]
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

    traverse(checks)
    if errors:
      raise ValueError('Invalid schema.\n\n' + '\n'.join(errors))
    return flat

  @staticmethod
  def _squeeze(checks: SchemaDict, n: int = 1) -> SchemaDict:
    """
    Squeeze redundancy out of schema checks.

    Examples
    --------
    >>> checks = {
    ...   Table('x'): {
    ...     Table(): [],
    ...     Table(): [Check.has_columns(['x'])]
    ...   },
    ...   Table('x'): {
    ...     Column('x'): [Check.not_null()],
    ...     Column('x'): [Check.unique()]
    ...   }
    ... }
    >>> Schema._squeeze(checks)
    {Table('x'):
      [Check.has_columns(columns=['x'], fill=False)],
    Table('x'):
      {Column('x'):
        [Check.not_null(), Check.unique()]}}
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

    checks = copylib.deepcopy(checks)
    for _ in range(n):
      drop_empty(checks)
      merge_equal_siblings(checks)
      unwrap_singletons(checks)
    return checks

  @staticmethod
  def _filter(checks: SchemaDict, target: Target) -> FlatSchemaDict:
    """
    Flatten, filter, and assign checks based on input data and name.

    Examples
    --------
    >>> checks = {
    ...   Table(): Check.has_columns(['x']),
    ...   Column(): Check.not_null(),
    ...   Column('x'): Check.unique(),
    ... }
    >>> Schema._filter(checks, Tables())
    {Table(): Check.has_columns(columns=['x'], fill=False),
    Column(): Check.not_null(),
    Column('x'): Check.unique()}
    >>> Schema._filter(checks, Table('a'))
    {Table('a'): Check.has_columns(columns=['x'], fill=False),
    Column(): Check.not_null(),
    Column('x'): Check.unique()}
    >>> Schema._filter(checks, Column('x', table='a'))
    {Column('x', table='a'): Check.not_null(),
    Column('x', table='a'): Check.unique()}
    >>> Schema._filter(checks, Column('x'))
    {Column('x'): Check.not_null(), Column('x'): Check.unique()}
    >>> Schema._filter(checks, Column())
    {Column(): Check.not_null()}
    """
    filtered = {}
    for key, check in Schema._flatten(checks).items():
      if key.equals(target):
        filtered[key] = check
      elif key.matches(target):
        filtered[target.copy()] = check
      elif key in target:
        filtered[key] = check
    return filtered

  @staticmethod
  def _expand_check(
    key: Target, check: Check, data: Data, name: Target
  ) -> SchemaDict:
    """
    Assign check based on input data and name.

    Assumes that `key` has already been filtered by :meth:`_filter`.

    Examples
    --------
    >>> checks = {
    ...   Table(): Check.has_columns(['x']),
    ...   Column(): Check.not_null(),
    ...   Column('x'): Check.unique(),
    ... }
    >>> checks = Schema._filter(checks, target=Tables())
    >>> dfs = {'a': pd.DataFrame({'x': [0], 'y': [1]})}
    >>> Schema._expand_check(Table(), Check.has_columns(['x']), dfs, Tables())
    {Table('a'): Check.has_columns(columns=['x'], fill=False)}
    >>> Schema._expand_check(Column(), Check.not_null(), dfs, Tables())
    {Column('x', table='a'): Check.not_null(),
    Column('y', table='a'): Check.not_null()}
    >>> Schema._expand_check(Column('x'), Check.unique(), dfs, Tables())
    {Column('x', table='a'): Check.unique()}
    >>> Schema._expand_check(Column('x'), Check.unique(), dfs['a'], Table())
    {Column('x'): Check.unique()}
    >>> Schema._expand_check(Column(), Check.not_null(), dfs['a']['x'], Column())
    {Column(): Check.not_null()}
    """
    # Matching keys replaced with equal keys by _filter
    if key.equals(name):
      return {key: check}
    # Only keys in target should remain
    assigned = {}
    if isinstance(key, (Column, Table)) and key.table is not None:
      assigned[key] = check
    elif isinstance(key, Table):
      assigned.update({Table(table): check for table in data})
    elif isinstance(key, Column):
      if isinstance(name, Table):
        columns = list(data) if key.column is None else [key.column]
        assigned.update(
          {Column(column, name.table): check for column in columns}
        )
      elif isinstance(name, Tables):
        for table in data:
          assigned.update(
            {
              Column(column, table): check
              for column in
              (data[table] if key.column is None else [key.column])
              if column in data[table]
            }
          )
    return assigned

  def __call__(
    self,
    data: Data = None,
    name: Target = None,
    target: Target = None,
    copy: bool = True,
    verbose: bool = False
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
    ...   Column(): Check.not_null(),
    ...   Column('x'): Check.unique()
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

    More complex example where elements are renamed, influencing choice of
    later wildcard checks.

    >>> df = pd.DataFrame(columns=['x', 'y'])
    >>> schema = Schema({
    ...   Table(): Check(lambda df: df.rename(columns={'x': 'z'}), test=False),
    ...   Column(): Check(lambda s: s.empty)
    ... })
    >>> report = schema(df)
    >>> report.results[1].target
    Column('z')
    >>> dfs = {'x': df}
    >>> schema = Schema({
    ...   Tables(): Check(lambda dfs: {'z': dfs['x']}, test=False),
    ...   Table(): Check(lambda df: df.empty)
    ... })
    >>> report = schema(dfs)
    >>> report.results[1].target
    Table('z')

    Example where elements are missing and thus skipped.

    >>> dfs = {'x': pd.DataFrame()}
    >>> check = Check(lambda df: df.empty)
    >>> check(dfs, target=Table('y'))
    Traceback (most recent call last):
      ...
    ValueError: Failed to extract data for Table('y') from data
    >>> schema = Schema({Table('y'): check})
    >>> schema(dfs).counts
    {'skip': 1}

    Example where input is modified in place.

    >>> dfs = {'x': pd.DataFrame(), 'y': pd.DataFrame()}
    >>> schema = Schema({
    ...   Tables(): Check.only_has_tables(['x'], drop=True),
    ...   Table(): Check(lambda df: df.empty)
    ... })
    >>> schema(dfs).counts
    {'pass': 2}
    """
    name = name or classify_data(data)()
    target = target or name
    inputs = extract_data(
      copylib.deepcopy(data) if copy else data,
      name=name,
      target=target
    )
    if copy:
      input = extract_data(data, name=name, target=target)[type(target)]
    else:
      input = inputs[type(target)]
    results = {}
    for okey, ocheck in Schema._filter(self.schema, target).items():
      expanded = Schema._expand_check(
        key=okey, check=ocheck, data=inputs[type(name)], name=name
      )
      for key, check in expanded.items():
        if verbose:
          print(f'{key}: {check}')
        # Pre-check existence of target in data to avoid Check.__call__() error
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
        # Run check
        result = check(inputs[type(name)], name=name, target=key)
        # Reassign new value
        output = result.output
        check_class = list(check.inputs.values())[0]
        try:
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
        except Exception as e:
          raise ValueError(f'Failed to set output of {check}.\n- error: {e}\n-output: {output}')
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
    ...   Table(): Check.has_column('x'),
    ...   Column('x'): Check.in_values([0, 1])
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
        if isinstance(subvalue, Check):
          subvalue = [subvalue]
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
    >>> x = Schema({Column('x'): Check(lambda s: s.notnull())})
    >>> y = Schema({Column('y'): Check(lambda s: s.notnull())})
    >>> (x + y).schema
    {Column('x'): Check.<lambda>(), Column('y'): Check.<lambda>()}
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

  def __add__(self, other: 'Report') -> 'Report':
    """
    Concatenate two reports.

    Parameters
    ----------
    other
      Report to concatenate to the end of the first report.

    Example
    -------
    >>> x = Schema({Column('x'): Check(lambda s: s.notnull())})
    >>> y = Schema({Column('y'): Check(lambda s: s.notnull())})
    >>> df = pd.DataFrame({'x': [0], 'y': [1]})
    >>> reports = x(df), y(df)
    >>> report = reports[0] + reports[1]
    >>> report
    Report(Table(), valid=True, counts={'pass': 2})
    >>> report.input is reports[0].input
    True
    >>> report.output is reports[1].output
    True
    """
    if not isinstance(other, self.__class__):
      return other + self
    if not self.target.equals(other.target):
      raise ValueError(
        'Cannot concatenate reports with different targets: ' +
        f'{self.target}, {other.target}'
      )
    return self.__class__(
      results=self.results + other.results,
      target=self.target,
      input=self.input,
      output=other.output
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
  def valid(self) -> bool:
    """
    Overall test result.

    - `True` if all result codes are 'pass' or 'skip'
    - `False` otherwise
    """
    return all(result.code in ['pass', 'skip'] for result in self.results)

  @property
  def empty(self) -> bool:
    """
    Whether report is empty.

    If :attr:`results` is empty or all result codes are 'skip'.
    """
    return (
      not self.results or
      all(result.code == 'skip' for result in self.results)
    )

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

  def project(self, data: Data, name: Target = None) -> Tuple['Report', 'Report']:
    """
    Project report onto data.

    Example
    -------
    >>> schema = Schema({
    ...   Column('x'): Check.not_null(),
    ...   Column('y'): Check.not_null()
    ... })
    >>> df = pd.DataFrame({'x': [0, pd.NA], 'y': [0, 1]})
    >>> report = schema(df)
    >>> report.project(df)
    (Report(Table(), valid=False, counts={'fail': 1, 'pass': 1}),
    Report(Table(), valid=True, counts={}))
    >>> report.project(df[['x']])
    (Report(Table(), valid=False, counts={'fail': 1}),
    Report(Table(), valid=True, counts={'pass': 1}))
    >>> rin, rout = report.project(df[['x']].iloc[:1])
    >>> rin.results[0].valid
    0    True
    Name: x, dtype: bool
    >>> rout.results[0].valid
    1   False
    Name: x, dtype: bool
    """
    name = name or classify_data(data)()
    ins, outs = [], []
    for result in self.results:
      try:
        inputs = extract_data(data, name=name, target=result.target)
      except ValueError:
        outs.append(result)
        continue
      if result.valid is None or isinstance(result.valid, bool):
        ins.append(result)
        continue
      input = inputs[type(result.target)]
      if isinstance(result.target, Tables):
        index = pd.Index(input.keys())
      elif isinstance(result.target, Table) and result.check.axis == 'column':
        index = input.columns
      elif isinstance(result.target, (Table, Column)) and result.check.axis == 'row':
        index = input.index
      else:
        assert False, 'Unexpected result format'
      mask = result.valid.index.isin(index)
      count = mask.sum()
      if count == 0:
        outs.append(result)
      elif count == mask.size:
        ins.append(result)
      else:
        base = dict(check=result.check, target=result.target)
        ins.append(Result(**base, valid=result.valid[mask]))
        outs.append(Result(**base, valid=result.valid[~mask]))
    return Report(results=ins, target=name), Report(results=outs, target=name)
