"""Schema conversion from Frictionless specs (https://specs.frictionlessdata.io)."""
from typing import Any, List, Optional, Union

from ..check import Check
from ..schema import Schema
from ..targets import Column, Table, Tables

# ---- Helpers ----


def _format_names(names: Union[str, List[str]] = None) -> Optional[List[str]]:
    """Format name lists."""
    if isinstance(names, str):
        return [names]
    return names


# ---- Conversion ----


def field_to_schema(field: dict, schema: dict = None) -> Schema:
    """
    Convert Frictionless Field to Schema.

    Parameters
    ----------
    field
        Frictionless Table Schema field descriptor
        (https://specs.frictionlessdata.io/table-schema/#field-descriptors).
    schema
        Frictionless Table Schema descriptor
        (https://specs.frictionlessdata.io/table-schema).
        If provided and `field['name']` is in `schema['primaryKey']`,
        then the field is required. Furthermore, if `schema['primaryKey']` only
        contains `field['name']`, then the field is unique.

    Examples
    --------
    >>> field = {'name': 'x', 'type': 'string', 'constraints': {'enum': ['a']}}
    >>> field_to_schema(field)
    Schema({Column('x'):
      [Check.parse_as_type(type='string'),
      Check.in_list(values=['a'])]})
    >>> field_to_schema(field, schema={'primaryKey': ['x', 'y']})
    Schema({Column('x'):
      [Check.not_null(),
      Check.parse_as_type(type='string'),
      Check.in_list(values=['a'])]})
    """
    checks = []
    primary_key = _format_names((schema or {}).get('primaryKey'))
    if field.get('constraints', {}).get('required', False) or (
        primary_key and field.get('name') in primary_key
    ):
        # Check for nulls before type coercion
        checks.append(Check.not_null())
    if field.get('type', 'any') != 'any':
        # Coerce to type before all other checks
        checks.append(Check.parse_as_type(type=field.get('type', 'any')))
    if field.get('constraints', {}).get('unique', False) or (
        primary_key and len(primary_key) == 1 and field.get('name') in primary_key
    ):
        checks.append(Check.unique())
    for k, v in field.get('constraints', {}).items():
        if k in ('minLength', 'min_length') and v is not None:
            checks.append(Check.length_greater_than_or_equal_to(v))
        elif k in ('maxLength', 'max_length') and v is not None:
            checks.append(Check.length_less_than_or_equal_to(v))
        elif k == 'minimum' and v is not None:
            checks.append(Check.greater_than_or_equal_to(v))
        elif k == 'maximum' and v is not None:
            checks.append(Check.less_than_or_equal_to(v))
        elif k == 'pattern' and v:
            checks.append(Check.matches_regex(v))
        elif k == 'enum' and v:
            checks.append(Check.in_list(v))
    return Schema(Schema._squeeze({Column(field.get('name')): checks}))


def schema_to_schema(
    schema: dict,
    require: Optional[bool] = False,
    order: Optional[bool] = False,
    strict: Optional[bool] = False,
) -> Schema:
    """
    Convert Frictionless Table Schema to Schema.

    Only internal, unnamed foreign keys are included
    (e.g. `{'fields': ['x'], 'reference': {'resource': '', 'fields': ['id']}}`).

    Parameters
    ----------
    schema
        Frictionless Table Schema descriptor
        (https://specs.frictionlessdata.io/table-schema).
    require
        Whether to check presence of non-required columns.
        If `None`, these columns, if missing, are filled with null values.
    order
        Whether to check column order.
        If `None`, columns are reordered to match.
    strict
        Whether to check presence of extra columns.
        If `None`, these columns are dropped.

    Examples
    --------
    >>> fields = [{'name': 'id'}, {'name': 'x'}]
    >>> schema = {'fields': fields}
    >>> schema_to_schema(schema)
    Schema({})
    >>> schema_to_schema(schema, require=True)
    Schema({Table():
      [Check.has_columns(columns=['id', 'x'], fill=False)]})
    >>> schema_to_schema(schema, require=None)
    Schema({Table():
      [Check.has_columns(columns=['id', 'x'], fill=True)]})
    >>> schema_to_schema(schema, order=True)
    Schema({Table():
      [Check.has_sorted_columns(columns=['id', 'x'], sort=False)]})
    >>> schema_to_schema(schema, order=None)
    Schema({Table():
      [Check.has_sorted_columns(columns=['id', 'x'], sort=True)]})
    >>> schema_to_schema(schema, strict=True)
    Schema({Table():
      [Check.only_has_columns(columns=['id', 'x'], drop=False)]})
    >>> schema_to_schema(schema, strict=None)
    Schema({Table():
      [Check.only_has_columns(columns=['id', 'x'], drop=True)]})

    With a primary key.

    >>> schema = {'fields': fields, 'primaryKey': ['id', 'x']}
    >>> schema_to_schema(schema)
    Schema({Table():
        [Check.has_columns(columns=['id', 'x'], fill=False)],
      Column('id'):
        [Check.not_null()],
      Column('x'):
        [Check.not_null()],
      Table():
        [Check.unique_rows(columns=['id', 'x'])]})

    With an internal foreign key.

    >>> foreign_key = {'fields': ['x'], 'reference': {'resource': '', 'fields': ['id']}}
    >>> schema = {'fields': fields, 'foreignKeys': [foreign_key]}
    >>> schema_to_schema(schema)
    Schema({Column('x'): [Check.in_column(column='id')]})
    """
    flow = {}
    checks = []
    if not schema['fields']:
        return Schema(flow)
    primary_key = _format_names(schema.get('primaryKey'))
    required = [
        field['name']
        for field in schema['fields']
        if (
            field.get('constraints', {}).get('required', False)
            or (primary_key and field['name'] in primary_key)
        )
    ]
    optional = [
        field['name'] for field in schema['fields'] if field['name'] not in required
    ]
    # Check column existence
    if require is True:
        checks.append(Check.has_columns(required + optional))
    elif require is False and required:
        checks.append(Check.has_columns(required))
    elif require is None:
        if required:
            checks.append(Check.has_columns(required))
        if optional:
            checks.append(Check.has_columns(optional, fill=True))
    # Check column order
    columns = [field['name'] for field in schema['fields']]
    if order is True:
        checks.append(Check.has_sorted_columns(columns, sort=False))
    elif order is None:
        checks.append(Check.has_sorted_columns(columns, sort=True))
    # Filter extra columns
    if strict is True:
        checks.append(Check.only_has_columns(columns, drop=False))
    elif strict is None:
        checks.append(Check.only_has_columns(columns, drop=True))
    # Check table column structure before columns
    flow[Table()] = checks
    # Column checks
    for field in schema['fields']:
        temp = field_to_schema(field, schema=schema)
        flow.update(temp.schema)
    # Check primary and foreign keys after columns
    checks = []
    if primary_key and len(primary_key) > 1:
        # Single-column primary keys are covered by field constraints
        checks.append(Check.unique_rows(primary_key))
    # Include only internal (unnamed) foreign keys
    for key in schema.get('foreignKeys', []):
        if not key['reference']['resource']:
            local = _format_names(key['fields'])
            foreign = _format_names(key['reference']['fields'])
            if len(local) == 1:
                # Use column check for single-column foreign keys
                flow[Column(local[0])] = [Check.in_column(foreign[0])]
            else:
                checks.append(Check.in_columns({x: y for x, y in zip(local, foreign)}))
    flow[Table()] = checks
    return Schema(Schema._squeeze(flow))


def resource_to_schema(resource: dict, **kwargs: Any) -> Schema:
    """
    Convert Frictionless Tabular Resource to Schema.

    Only internal foreign keys are included.

    Parameters
    ----------
    schema
        Frictionless Tabular Resource descriptor
        (https://specs.frictionlessdata.io/tabular-data-resource).
    **kwargs
        Optional arguments to :func:`schema_to_schema`.

    Examples
    --------
    >>> fields = [{'name': 'id'}, {'name': 'x'}]
    >>> resource = {'name': 'tab', 'schema': {'fields': fields}}
    >>> resource_to_schema(resource)
    Schema({})
    >>> resource_to_schema(resource, require=True)
    Schema({Table('tab'):
      [Check.has_columns(columns=['id', 'x'], fill=False)]})

    With an internal foreign key.

    >>> foreign_key = {
    ...     'fields': ['x'],
    ...     'reference': {'resource': 'tab', 'fields': ['id']}
    ... }
    >>> resource = {
    ...     'name': 'tab',
    ...     'schema': {'fields': fields, 'foreignKeys': [foreign_key]}
    ... }
    >>> resource_to_schema(resource)
    Schema({Table('tab'): [Check.in_columns(columns={'x': 'id'})]})
    """
    flow = schema_to_schema(resource['schema'], **kwargs).schema
    # Internal named foreign keys only
    checks = []
    for key in resource['schema'].get('foreignKeys', []):
        if (
            key['reference']['resource']
            and key['reference']['resource'] == resource['name']
        ):
            local = _format_names(key['fields'])
            foreign = _format_names(key['reference']['fields'])
            checks.append(Check.in_columns({x: y for x, y in zip(local, foreign)}))
    flow[Table()] = checks
    flow = {Table(resource['name']): flow}
    return Schema(Schema._squeeze(flow, n=2))


def package_to_schema(
    package: dict,
    require: Optional[bool] = False,
    order: Optional[bool] = False,
    strict: Optional[bool] = False,
    columns: dict = None,
) -> Schema:
    """
    Convert Frictionless Tabular Data Package to Schema.

    Parameters
    ----------
    schema
        Frictionless Tabular Data Package descriptor
        (https://specs.frictionlessdata.io/tabular-data-package).
    require
        Whether to check presence of non-required tables.
        If `None`, tables, if missing, are filled with null values.
    order
        Whether to check table order.
        If `None`, tables are reordered to match.
    strict
        Whether to check presence of extra tables.
        If `None`, these tables are dropped.
    columns
        Column-level arguments passed to :func:`schema_to_schema`.

    Examples
    --------
    >>> schema = {'fields': [{'name': 'id'}]}
    >>> resources = [{'name': 'x', 'schema': schema}, {'name': 'y', 'schema': schema}]
    >>> package = {'resources': resources}
    >>> package_to_schema(package)
    Schema({})
    >>> package_to_schema(package, require=True)
    Schema({Tables(): [Check.has_tables(tables=['x', 'y'], fill=False)]})
    >>> package_to_schema(package, order=True)
    Schema({Tables(): [Check.has_sorted_tables(tables=['x', 'y'], sort=False)]})
    >>> package_to_schema(package, order=None)
    Schema({Tables(): [Check.has_sorted_tables(tables=['x', 'y'], sort=True)]})
    >>> package_to_schema(package, strict=True)
    Schema({Tables(): [Check.only_has_tables(tables=['x', 'y'], drop=False)]})
    >>> package_to_schema(package, strict=None)
    Schema({Tables(): [Check.only_has_tables(tables=['x', 'y'], drop=True)]})

    With a foreign key.

    >>> foreign_key = {
    ...     'fields': ['id'],
    ...     'reference': {'resource': 'x', 'fields': ['id']}
    ... }
    >>> schema_with_key = {'fields': [{'name': 'id'}], 'foreignKeys': [foreign_key]}
    >>> package['resources'][1]['schema'] = schema_with_key
    >>> package_to_schema(package)
    Schema({Column('id', table='y'): [Check.in_foreign_column(table='x', column='id')]})
    """
    flow = {}
    if not package['resources']:
        return Schema(flow)
    checks = []
    required = [resource['name'] for resource in package['resources']]
    # Check table existence
    if require is True:
        checks.append(Check.has_tables(required))
    elif require is None:
        checks.append(Check.has_tables(required, fill=True))
    # Check table order
    if order is True:
        checks.append(Check.has_sorted_tables(required))
    elif order is None:
        checks.append(Check.has_sorted_tables(required, sort=True))
    # Filter extra tables
    if strict is True:
        checks.append(Check.only_has_tables(required))
    elif strict is None:
        checks.append(Check.only_has_tables(required, drop=True))
    # Check table structure before tables
    flow[Tables()] = checks
    # Table checks
    for resource in package['resources']:
        flow.update(resource_to_schema(resource, **(columns or {})).schema)
    # Cross-table foreign keys only at the very end
    for resource in package['resources']:
        tasks = []
        for key in resource['schema'].get('foreignKeys', []):
            if (
                key['reference']['resource']
                and key['reference']['resource'] != resource['name']
            ):
                local = _format_names(key['fields'])
                foreign = _format_names(key['reference']['fields'])
                if len(local) == 1:
                    # Use a column check for single-column foreign key
                    flow[Column(local[0], table=resource['name'])] = [
                        Check.in_foreign_column(
                            table=key['reference']['resource'], column=foreign[0]
                        )
                    ]
                else:
                    tasks.append(
                        Check.in_foreign_columns(
                            table=key['reference']['resource'],
                            columns={x: y for x, y in zip(local, foreign)},
                        )
                    )
        flow[Table(resource['name'])] = tasks
    return Schema(Schema._squeeze(flow, n=2))
