# Allow the use of copy= function argument
import copy as copylib
from typing import Any, Dict, Hashable, Iterable, List, Tuple, Union

import pandas as pd

from .check import Check, Result, Report
from .targets import Target, Column, Table, Tables

# ---- Schema helpers ----

SchemaDict = Dict[
  Target,
  Union[
    Check,
    Iterable[Check],
    Dict[Union[Column, Table], Union[Check, Iterable[Check]]]
  ]
]
FlatSchemaDict = Dict[Target, Check]

def _flatten_schema(schema: SchemaDict) -> Tuple[FlatSchemaDict, List[str]]:
  errors = []
  flattened = {}

  def flatten_value(key: Target, value: Any) -> None:
    if isinstance(value, Check):
      save_check(key, value)
    elif isinstance(value, Iterable):
      for subvalue in value:
        if isinstance(subvalue, Check):
          save_check(copylib.copy(key), subvalue)
        else:
          errors.append(f'{prefix} {subvalue}: Not a Check')
    else:
      errors.append(f'{prefix} {value}: Not a Check or Iterable[Check]')

  def save_check(key: Target, check: Check) -> None:
    if check.scope != key.scope:
      errors.append(f'{prefix} {check}: Check has wrong scope ({check.scope})')
    # required = [scope for scope, required in check.context.items() if required]
    # if required:
    #   required = max_scope(*required)
    # if required in ('table', 'tables') and isinstance(key, Column) and key.column is None:
    #   errors.append(
    #     f"{prefix} {check}: Column name is required since check depends on {required}"
    #   )
    # if required == 'tables' and isinstance(key, (Column, Table)) and key.table is None:
    #   errors.append(
    #     f"{prefix} {check}: Table name is required since check depends on {required}"
    #   )
    else:
      flattened[key] = check

  for key, value in schema.items():
    prefix = f'[{key}]'
    if not isinstance(key, (Column, Table, Tables)):
      errors.append(f'{prefix} Not a Column, Table, or Tables')
    elif not isinstance(value, dict):
      flatten_value(key, value)
    elif not isinstance(key, Table):
      errors.append(f'{prefix} Cannot have nested schemas')
    else:
      for subkey, subvalue in value.items():
        prefix = f'[{key}] [{subkey}]'
        if isinstance(subkey, Table):
          if subkey.table is not None and subkey != key.table:
            errors.append(f'{prefix} Cannot nest schema for {subkey} in {key}')
          else:
            flatten_value(Table(key.table), subvalue)
        elif not isinstance(subkey, Column):
          errors.append(f'{prefix} Cannot be a nested schema')
        elif isinstance(subvalue, dict):
          errors.append(f'{prefix} Cannot have nested schemas')
        else:
          try:
            column_key = Column(subkey.column, table=key.table)
          except ValueError as error:
            errors.append(f'{prefix} {error}')
            continue
          flatten_value(column_key, subvalue)
  return flattened, errors


# ---- Schema ----

class Schema:
  """
  Schema({
    Tables(): [*checks],
    Table('A'): [*checks],
    Table('A'): {
      Column('a'): Check,
      Column('b'): [*checks]
    },
    Column('a', table='A'): [*checks]
  })
  """
  def __init__(self, schema: SchemaDict) -> None:
    if not isinstance(schema, dict):
      raise TypeError(f'Schema must be a dictionary, not a {type(schema)}')
    if not schema:
      raise ValueError(f'Schema cannot be empty')
    _, errors = _flatten_schema(schema)
    if errors:
      raise ValueError('Invalid schema.\n\n' + '\n'.join(errors))
    self.schema = schema
    # seen = []
    # unique_keys = []
    # for key in flat_schema:
    #   id = (key.__class__, key.__dict__)
    #   if id not in seen:
    #     seen.append(id)
    #     unique_keys.append(key.__class__(**key.__dict__))
    # scopes = {'column', 'table', 'tables'}
    # for key in unique_keys:
    #   scopes = scopes.intersection(key.scopes)
    # if not scopes:
    #   errors.append(
    #     ('\n\n' if errors else '') +
    #     ("Keys have incompatible scopes.\n\n") +
    #     '\n'.join(f'[{key}] {key.scopes}' for key in unique_keys)
    #   )

  def validate(
    self,
    column: pd.Series = None,
    table: pd.DataFrame = None,
    tables: Dict[Hashable, pd.DataFrame] = None,
    *,
    target: Target = None,
    copy: bool = True
  ) -> List[Result]:
    context = {Column: column, Table: table, Tables: tables}
    if all(value is None for value in context.values()):
      raise ValueError('Provide at least one of column, table, or tables')
    if target and not isinstance(target, tuple(context)):
      raise ValueError(f'Target must an instance of Column, Table, or Tables')
    for cls, value in context.items():
      if value is not None and target is None:
        target = cls()
      elif value is None and isinstance(target, cls):
        raise ValueError(f'Provide a value for {target} ({cls.__name__.lower()})')
    input = context[target.__class__]
    if copy:
      input = copylib.deepcopy(input)
    # if column is None and table is None and tables is None:
    #   raise ValueError('Provide at least one of column, table, or tables')
    # if target is None:
    #   if column is not None:
    #     target = Column()
    #   elif table is not None:
    #     target = Table()
    #   elif tables is not None:
    #     target = Tables()
    # if isinstance(target, Column) and column is None:
    #   raise ValueError(f'Provide a value for {target} (column)')
    # elif isinstance(target, Table) and table is None:
    #   raise ValueError(f'Provide a value for {target} (table)')
    # elif isinstance(target, Tables) and tables is None:
    #   raise ValueError(f'Provide a value for {target} (tables)')
    # else:
    #   raise ValueError(f'Target must be a Column, Table, or Tables')

    # input = context[target.__class__]
    results = {}
    for key, check in _flatten_schema(self.schema)[0].items():
      if not target.includes(key):
        continue
      # Load data for check
      args = {'column': column, 'table': table, 'tables': tables}
      if isinstance(target, Tables) and not isinstance(key, Tables):
        if not key.table in args['tables']:
          message = f'Table {key.table} not in tables'
          results[key] = Result(check, target=key, skip=message)
          continue
        args['table'] = args['tables'][key.table]
      if isinstance(target, (Table, Tables)) and isinstance(key, Column):
        if not key.column in args['table']:
          table_str = '' if key.table is None else f' {key.table}'
          message = f'Column {key.column} not in table{table_str}'
          results[key] = Result(check, target=key, skip=message)
          continue
        args['column'] = args['table'][key.column]
      # Check completeness of positional arguments
      missing = [
        scope for scope, required in check.scopes.items()
        if required and args[scope] is None
      ]
      if missing:
        message = f'Required inputs {missing} not provided'
        results[key] = Result(check, target=key, skip=message)
        continue
      # Run check
      result = check(
        value=args[check.scope],
        table=args['table'],
        tables=args['tables'],
        target=key
      )
      # Reassign new value
      if result.output is not None and result.output is not args[check.scope]:
        if check.scope == 'tables':
          tables = result.output
        elif check.scope == 'table':
          if isinstance(target, Table):
            table == result.output
          else:
            tables[key.table] = result.output
        elif check.scope == 'column':
          if isinstance(target, Column):
            column == result.output
          elif isinstance(target, Table):
            table[key.column] = result.output
          else:
            tables[key.table][key.column] = result.output
      results[key] = result
      # Process results
      # if type(key) is type(target):
      #   data = input
      # elif isinstance(key, Column):
      #   if isinstance(target, Table):
      #     if key.column not in input:
      #       results[key] = f'Column {key.column} not in input'
      #       break
      #     data = input[key.column]
      #   elif isinstance(target, Tables):
      #     if key.table not in input:
      #       results[key] = f'Table {key.table} not in input'
      #       break
      #     if key.column not in input[key.table]:
      #       results[key] = f'Column {key.column} not in input'
      #       break
      #     data = input[key.table][key.column]
      # elif isinstance(key, Table):
      #   if key.table not in input[key.table]:
      #     results[key] = f'Table {key.table} not in input'
      #     break
      #   data = input[key.table]
    output = {'column': column, 'table': table, 'tables': tables}[target.scope]
    return Report(results.values(), target=target, input=input, output=output)
