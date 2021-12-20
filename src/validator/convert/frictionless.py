from typing import Dict, List, Optional, TYPE_CHECKING, Union
import typing

from .. import checks
from ..check import Check
from ..targets import Column, Table, Tables

if TYPE_CHECKING:
  from frictionless import Field, Schema, Resource, Package

def _format_names(names: Union[str, List[str]] = None) -> Optional[List[str]]:
  if isinstance(names, str):
    return [names]
  return names

def field_to_schema(field: 'Field', schema: 'Schema' = None) -> Dict[Column, List[Check]]:
  tasks = []
  primary_key = _format_names(schema.primary_key)
  # Check for nulls before type coercion
  if (
    field.constraints.get('required', False) or
    (primary_key and field.name in primary_key)
  ):
    tasks.append(checks.column.not_null())
  if field.type != 'any':
    tasks.append(checks.column.parse_as_type(type=field.type))
  if (
    field.constraints.get('unique', False) or
    (primary_key and len(primary_key) == 1 and field.name in primary_key)
  ):
    tasks.append(checks.column.unique())
  for k, v in field.constraints.items():
    if k in ('minLength', 'min_length') and v is not None:
      tasks.append(checks.column.length_greater_than_or_equal_to(v))
    elif k in ('maxLength', 'max_length') and v is not None:
      tasks.append(checks.column.length_less_than_or_equal_to(v))
    elif k == 'minimum' and v is not None:
      tasks.append(checks.column.greater_than_or_equal_to(v))
    elif k == 'maximum' and v is not None:
      tasks.append(checks.column.less_than_or_equal_to(v))
    elif k == 'pattern' and v:
      tasks.append(checks.column.matches_regex(v))
    elif k == 'enum' and v:
      tasks.append(checks.column.in_list(v))
  return {Column(field.name): tasks}

def schema_to_schema(schema: 'Schema', strict: bool = True) -> Dict[Union[Table, Column], List[Check]]:
  flow = {}
  # Table checks (before columns)
  tasks = []
  primary_key = _format_names(schema.primary_key)
  required = [
    field.name for field in schema.fields
    if (
      field.constraints.get('required', False) or
      (primary_key and field.name in primary_key)
    )
  ]
  optional = [
    field.name for field in schema.fields if field.name not in required
  ]
  if strict:
    tasks.append(checks.table.has_columns(required + optional))
    tasks.append(checks.table.only_has_columns(required + optional, drop=False))
  else:
    if required:
      tasks.append(checks.table.has_columns(required))
    if optional:
      tasks.append(checks.table.has_columns(optional, fill=True))
    tasks.append(checks.table.only_has_columns(required + optional, drop=True))
  tasks.append(checks.table.has_sorted_columns(schema.field_names, sort=not strict))
  flow[Table()] = tasks
  # Column checks
  for field in schema.fields:
    flow.update(field_to_schema(field, schema=schema))
  # Table checks (after columns)
  tasks = []
  if primary_key and len(primary_key) > 1:
    tasks.append(checks.table.unique_rows(primary_key))
  # Internal unnamed foreign keys only
  for key in schema.foreign_keys:
    if not key['reference']['resource']:
      local = _format_names(key['fields'])
      foreign = _format_names(key['reference']['fields'])
      if len(local) == 1:
        flow[Column(local[0])] = [checks.column.in_column(foreign[0])]
      else:
        tasks.append(
          checks.table.in_columns({x: y for x, y in zip(local, foreign)})
        )
  if tasks:
    flow[Table()] = tasks
  return flow

def resource_to_schema(resource: 'Resource', strict: bool = True) -> Dict[Table, Dict[Union[Table, Column], List[Check]]]:
  flow = schema_to_schema(resource.schema, strict=strict)
  # Internal named foreign keys only
  tasks = []
  for key in resource.schema.foreign_keys:
    if key['reference']['resource'] and key['reference']['resource'] == resource.name:
      local = _format_names(key['fields'])
      foreign = _format_names(key['reference']['fields'])
      tasks.append(
        checks.table.in_columns({x: y for x, y in zip(local, foreign)})
      )
  if tasks:
    flow[Table()] = tasks
  return {Table(resource.name): flow}

def package_to_schema(package: 'Package', strict: bool = True) -> Dict[Union[Tables, Table], Dict[Union[Table, Column], List[Check]]]:
  flow = {}
  # Tables (before table checks)
  tasks = []
  required = [resource.name for resource in package.resources]
  tasks.append(checks.tables.has_tables(required, fill=not strict))
  tasks.append(checks.tables.only_has_tables(required, drop=not strict))
  tasks.append(checks.tables.has_sorted_tables(package.resource_names, sort=not strict))
  flow[Tables()] = tasks
  # Table checks
  for resource in package.resources:
    flow.update(resource_to_schema(resource, strict=strict))
  # Cross-table foreign keys only
  for resource in package.resources:
    tasks = []
    for key in resource.schema.foreign_keys:
      if key['reference']['resource'] and key['reference']['resource'] != resource.name:
        local = _format_names(key['fields'])
        foreign = _format_names(key['reference']['fields'])
        if len(local) == 1:
          flow[Column(local[0], table=resource.name)] = checks.column.in_foreign_column(
            table=key['reference']['resource'], column=foreign[0]
          )
        else:
          tasks.append(
            checks.table.in_foreign_columns(
              table=key['reference']['resource'],
              columns={x: y for x, y in zip(local, foreign)}
            )
          )
    if tasks:
      flow[Table(resource.name)] = tasks
  return flow
