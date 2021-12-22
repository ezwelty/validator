import inspect
from inspect import Parameter, Signature
from typing import Any, Callable, Dict, Hashable, Iterable, List, Literal, Optional, Sequence, Union

import makefun
import pandas as pd

from .helpers import ARGUMENT_SCOPES, filter_kwargs, RANKS, Scope, superscopes
from .targets import Column, Table, Tables, Target


# ---- Helpers ----

def parse_check_function(
  fn: Callable, args: Dict[str, Scope] = None, kwargs: Dict[str, Any] = None
) -> dict:
  """
  Parse properties of a check function from its signature.

  Examples:
    >>> def fn(s, df, *, x, y=None):
    ...   pass
    >>> parse_check_function(fn, kwargs={'x': 1, 'y': 2})
    {'args': {'s': 'column', 'df': 'table'}, 'kwargs': {'x': 1, 'y': 2}}
    >>> def fn(df, dfs=None):
    ...   pass
    >>> parse_check_function(fn)
    {'args': {'df': 'table', 'dfs': 'tables'}, 'kwargs': {}}
  """
  signature = inspect.signature(fn)
  # Extract positional argument names from function
  fn_args = [
    name for name, param in signature.parameters.items()
    if param.kind == param.POSITIONAL_OR_KEYWORD
  ]
  if not fn_args:
    raise ValueError(f'{fn} must have at least one positional argument.')
  # Ensure argument scopes are valid
  if args:
    invalid = [scope for scope in args.values() if scope not in RANKS]
    if invalid:
      raise ValueError(f'Invalid argument scopes {invalid}.')
    arg_scopes = {**ARGUMENT_SCOPES, **args}
  else:
    arg_scopes = ARGUMENT_SCOPES
  # Map argument names to scopes
  args = {}
  invalid = []
  for name in fn_args:
    if name in arg_scopes:
      scope = arg_scopes[name]
      if scope in args.values():
        assigned = [x for x in args if args[x] == scope][0]
        raise ValueError(
          f"Arguments {assigned} and {name} map to the same scope '{scope}'."
        )
      args[name] = scope
    else:
      invalid.append(name)
  if invalid:
    raise ValueError(
      f"Could not determine the scopes of {fn} arguments {invalid}."
      " Supply their scope or name positional arguments"
      " 's' / 'column', 'df' / 'table', or 'dfs' / 'tables'."
    )
  # Check that arguments are parents of first argument
  child = list(args.values())[0]
  parents = superscopes(child)
  not_parents = {
    name: scope for i, (name, scope) in enumerate(args.items())
    if i > 0 and scope not in parents
  }
  if not_parents:
    raise ValueError(
      f"Arguments {list(not_parents)} ({list(not_parents.values())})"
      f" are not parents of {list(args)[0]} ({list(args.values())[0]})."
      f" Their scope must be one of {parents}."
    )
  # Params
  kwargs = kwargs or {}
  fn_kwargs = {
    name: param.default is inspect._empty
    for name, param in signature.parameters.items()
    if param.kind == param.KEYWORD_ONLY
  }
  unexpected = [key for key in kwargs if key not in fn_kwargs]
  if unexpected:
    raise ValueError(f"Unexpected keyword arguments {unexpected}.")
  missing = [
    key for key, required in fn_kwargs.items() if required and key not in kwargs
  ]
  if missing:
    raise ValueError(f"Missing required keyword arguments {missing}.")
  # Return
  # TODO: Inspect Tuple argument types
  # TODO: Inspect other return types
  # return_type = signature.return_annotation
  # if return_type is inspect._empty:
  #   if transform is None:
  #     raise ValueError(
  #       f"Could not determine whether {fn} returns a new value of its input."
  #       " Supply transform (True or False) or add a return type annotation."
  #     )
  # else:
  #   if return_type is tuple or typing.get_origin(return_type) is tuple:
  #     if transform is False:
  #       raise ValueError(
  #         f"Supplied transform=False but {fn} returns a tuple"
  #       )
  #     transform = True
  #   else:
  #     if transform is True:
  #       raise ValueError(
  #         f"Supplied transform=True but {fn} does not return a tuple"
  #       )
  #     transform = False
  return {"args": args, "kwargs": kwargs}

Value = Union[pd.Series, pd.DataFrame, Dict[Hashable, pd.DataFrame]]
ParentValue = Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]]
Valid = Union[bool, pd.Series]
# CheckContext = Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]]

class Check:

  def __init__(
    self,
    fn: Callable,
    args: Dict[str, Scope] = None,
    kwargs: Dict[str, Any] = None,
    requires: Union[Column, Table, Sequence[Union[Column, Table]]] = None,
    message: str = None,
    name: str = None,
    severity: Literal['error', 'warning'] = 'error',
    axis: Literal['row', 'column', 'table'] = None
  ) -> None:
    # TODO: Check that requires= is consistent with positional arguments
    parsed = parse_check_function(fn, args=args, kwargs=kwargs)
    self.fn = fn
    self.args: Dict[str, Scope] = parsed['args']
    self.kwargs: Dict[str, Any] = parsed['kwargs']
    self.message = message
    self.name = name or fn.__name__
    self.severity = severity
    if axis is not None:
      if (
        (self.scope == 'tables' and axis not in {'table'}) or
        (self.scope == 'table' and axis not in {'row', 'column'}) or
        (self.scope == 'column' and axis not in {'row'})
      ):
        raise ValueError(f'Unsupported axis {axis} for {self.scope} check')
    self.axis = axis or ('table' if self.scope == 'tables' else 'row')
    # Requires
    if requires is None:
      requires = []
    elif isinstance(requires, (Column, Table)):
      requires = [requires]
    else:
      requires = list(requires)
    for r in requires:
      if (
        isinstance(r, Tables) or
        (isinstance(r, Table) and r.table is None) or
        (isinstance(r, Column) and r.table is None and r.column is None)
      ):
        raise ValueError(
          'Require {r} by using a {r.scope} positional argument'
        )
      if isinstance(r, Table) and not self.scopes.get('tables', False):
          raise ValueError('Cannot require {r} without a tables argument')
      if isinstance(r, Column):
        if r.table is None and not self.scopes.get('table', False):
          raise ValueError(f'Cannot require {r} without a table argument')
        elif r.table is not None and not self.scopes.get('tables', False):
          raise ValueError(f'Cannot require {r} without a tables argument')
    self.requires: List[Union[Column, Table]] = requires

  @property
  def scope(self) -> Scope:
    return list(self.args.values())[0]

  @property
  def scopes(self) -> Dict[Scope, bool]:
    signature = inspect.signature(self.fn)
    return {
      scope: signature.parameters[name].default is inspect._empty
      for name, scope in self.args.items()
    }

  def __repr__(self) -> str:
    args = ', '.join(
      f'{key}={value}' for key, value in self.kwargs.items()
      if value is not None
    )
    return f'Check[{self.scope}].{self.name}({args})'

  # def __call__(self, value: Value, **parents: ParentValue) -> Result:
  #   signature = inspect.signature(self.fn)
  #   parent_args = [
  #     parents[name] for name, param in list(signature.parameters.items())[1:]
  #     if param.kind == param.POSITIONAL_OR_KEYWORD
  #     and name in context
  #   ]
  #   return self.fn(value, *parent_args, **self.kwargs)

  def __call__(
    self,
    value: Value,
    table: pd.DataFrame = None,
    tables: Dict[Hashable, pd.DataFrame] = None,
    target: Target = None
  ) -> 'Result':
    if target and target.scope != self.scope:
        raise ValueError(
          f'Target {target} has wrong scope for {self.scope} check'
        )
    # Construct positional arguments
    args = {}
    for i, (name, scope) in enumerate(self.args.items()):
      if i == 0:
        args[name] = value
      elif scope == 'table':
        if table is None and self.scopes[scope]:
          raise ValueError(f'Required {scope} not provided')
        args[name] = table
      elif scope == 'tables':
        if tables is None and self.scopes[scope]:
          raise ValueError(f'Required {scope} not provided')
        args[name] = tables
    # Execute function
    try:
      result = self.fn(**args, **self.kwargs)
    except Exception as error:
      return Result(self, target=target, valid=None, input=value, error=error)
    if isinstance(result, tuple):
      valid, output = result
    else:
      valid, output = result, None
    # Check result
    try:
      if isinstance(valid, dict):
        valid = pd.Series(valid, dtype='boolean')
      elif isinstance(valid, pd.Series) and not pd.api.types.is_bool_dtype(valid):
        valid = valid.astype('boolean', copy=False)
    except TypeError:
      raise ValueError(
        f'Result of {self.fn} could not be coerced to a boolean Series'
      )
    return Result(self, target=target, valid=valid, input=value, output=output)

  # def __call__(
  #   self,
  #   value: CheckValue,
  #   **context: CheckContext
  # ) -> Union[CheckResult, Tuple[CheckResult, CheckValue]]:
  #   # TODO: Check input type against check type
  #   sig = inspect.signature(self.fn)
  #   params = sig.parameters
  #   context_args = [
  #     context[name] for name, param in list(params.items())[1:]
  #     if param.kind == param.POSITIONAL_OR_KEYWORD
  #     and name in context
  #   ]
  #   return self.fn(value, *context_args, **self.kwargs)

  # def _column(self, s: pd.Series, df: pd.DataFrame = None, dfs: Dict[Hashable, pd.DataFrame] = None) -> tuple[CheckResult, pd.Series]:
  #   # Check argument types
  #   assert isinstance(s, pd.Series)
  #   assert df is None or isinstance(df, pd.DataFrame)
  #   assert dfs is None or (isinstance(dfs, dict) and all(isinstance(x, pd.DataFrame) for x in dfs.values()))
  #   # Run check
  #   result = self(s, df=df, dfs=dfs)
  #   # Parse and check result type
  #   if isinstance(result, tuple):
  #     valid, output = result
  #     # TODO: Allow output=None
  #     assert isinstance(output, pd.Series) and output.index.equals(s.index)
  #   else:
  #     valid, output = result, s
  #   assert isinstance(valid, (bool, pd.Series))
  #   if isinstance(valid, pd.Series):
  #     assert valid.index.equals(s.index)
  #     if not pd.api.types.is_bool_dtype(valid):
  #       try:
  #         valid = valid.astype('boolean', copy=False)
  #       except TypeError:
  #         raise ValueError(f'Result could not be coerced to boolean from type {valid.dtype}')
  #   # Build report
  #   report = CheckResult(check=self)
  #   if isinstance(valid, bool):
  #     report.passed = valid
  #     report.values = None if report.passed else s
  #   else:
  #     report.passed = valid.all()
  #     report.values = None if report.passed else s[~valid]
  #   return report, output

  # def _table(self, df: pd.DataFrame, dfs: Dict[Hashable, pd.DataFrame] = None) -> Tuple[CheckResult, pd.DataFrame]:
  #   # Check argument types
  #   assert isinstance(df, pd.DataFrame)
  #   assert dfs is None or (isinstance(dfs, dict) and all(isinstance(x, pd.DataFrame) for x in dfs.values()))
  #   # Run check
  #   result = self(df, dfs=dfs)
  #   # Parse and check result type
  #   if isinstance(result, tuple):
  #     valid, output = result
  #     # TODO: Allow output=None
  #     assert isinstance(output, pd.DataFrame) and output.index.equals(df.index)
  #   else:
  #     valid, output = result, df
  #   assert isinstance(valid, (bool, pd.Series))
  #   if isinstance(valid, pd.Series):
  #     # TODO: Allow valid indexed by column
  #     assert valid.index.equals(df.index)
  #     if not pd.api.types.is_bool_dtype(valid):
  #       try:
  #         valid = valid.astype('boolean', copy=False)
  #       except TypeError:
  #         raise ValueError(f'Result could not be coerced to boolean from type {valid.dtype}')
  #   # Build report
  #   report = CheckResult(check=self)
  #   if isinstance(valid, bool):
  #     report.passed = valid
  #   else:
  #     report.passed = valid.all()
  #   return report, output

  # def _tables(self, dfs: Dict[Hashable, pd.DataFrame]) -> Tuple[CheckResult, Dict[Hashable, pd.DataFrame]]:
  #   # Check argument types
  #   assert isinstance(dfs, dict) and all(isinstance(x, pd.DataFrame) for x in dfs.values())
  #   # Run check
  #   result = self(dfs)
  #   # Parse and check result type
  #   if isinstance(result, tuple):
  #     valid, output = result
  #     # TODO: Allow output=None
  #     assert (
  #       isinstance(output, dict) and
  #       all(
  #         isinstance(x, pd.DataFrame) and
  #         name in dfs and
  #         x.index.equals(dfs[name].index)
  #         for name, x in output.items()
  #       )
  #     )
  #   else:
  #     valid, output = result, dfs
  #   # TODO: Allow valid indexed by table
  #   assert isinstance(valid, bool)
  #   # Build report
  #   return CheckResult(check=self, passed=valid), output

  # @classmethod
  # def not_null(cls, error: str = 'Missing value', **kwargs) -> 'Check':
  #   return cls(not_null, error=error, **kwargs)

  # @classmethod
  # def greater_than_or_equal_to(cls, min: Any, error: str = 'Value < {min}', **kwargs) -> 'Check':
  #   return cls(greater_than_or_equal_to, min=min, error=error, **kwargs)


class Result:

  def __init__(
    self,
    check: Check,
    target: Target,
    input: Value = None,
    output: Value = None,
    valid: Valid = None,
    error: Exception = None,
    skip: str = None
  ) -> None:
    self.check = check
    self.target = target
    self.valid = valid
    self.input = input
    self.output = output
    self.error = error
    self.skip = skip

  def __repr__(self):
    kwargs = ', '.join(
      f'{key}={value}' for key, value in self.__dict__.items()
      if value is not None
    )
    return f'Result({kwargs})'

  @property
  def status(self) -> Literal['pass', 'fail', 'error', 'skip']:
    if self.error is not None:
      return 'error'
    if (
      (self.valid is None and self.error is None) or
      (isinstance(self.valid, pd.Series) and self.valid.empty)
    ):
      return 'skip'
    if (
      self.valid is True or
      (isinstance(self.valid, pd.Series) and self.valid.all())
    ):
      return 'pass'
    if (
      self.valid is False or
      (isinstance(self.valid, pd.Series) and not self.valid.all())
    ):
      return 'fail'
    assert False

  @property
  def message(self) -> Optional[str]:
    status = self.status
    if status == 'fail' and self.check.message is not None:
      # return self.check.error.format(**self.check.kwargs)
      return eval(f"f'{self.check.message}'", None, self.check.kwargs)
    if status == 'error':
      return str(self.error)
    if status == 'skip' and self.skip is not None:
      return self.skip

  @property
  def axis(self) -> Optional[Literal['row', 'column', 'table']]:
    if self.valid is None or isinstance(self.valid, bool):
      return None
    return self.check.axis

  @property
  def table(self) -> Optional[Union[Hashable, List[Hashable]]]:
    return (
      (self.axis == 'table' and list(self.valid[~self.valid].index)) or
      (isinstance(self.target, (Table, Column)) and self.target.table) or
      None
    )

  @property
  def column(self) -> Optional[Union[Hashable, List[Hashable]]]:
    return (
      (self.axis == 'column' and list(self.valid[~self.valid].index)) or
      (isinstance(self.target, Column) and self.target.column) or
      None
    )

  @property
  def row(self) -> Optional[List[Hashable]]:
    return (
      (self.axis == 'row' and list(self.valid[~self.valid].index)) or
      None
    )

  @property
  def value(self) -> Optional[List[Hashable]]:
    return (
      (self.axis == 'row' and isinstance(self.target, Column) and list(self.input[~self.valid])) or
      # (self.axis == 'row' and isinstance(self.target, Table) and list(self.input[~self.valid].to_dict('records'))) or
      None
    )

  def to_dict(self) -> dict:
    return {
      'table': self.table,
      'column': self.column,
      'row': self.row,
      'value': self.value,
      'check': self.check,
      'severity': self.check.severity,
      'status': self.status,
      'message': self.message
    }

  # @property
  # def scope(self) -> str:
  #   if not self.table:
  #     return 'tables'
  #   if not self.columns and not self.rows:
  #     return 'table'
  #   if not self.columns:
  #     return 'rows'
  #   # NOTE: Currently a single column only past this point
  #   if not self.rows:
  #     return 'columns'
  #   return 'cells'

  # def as_dict(self) -> dict:
  #   d = {'scope': self.scope, 'check': self.check, 'message': self.message}
  #   if self.scope != 'tables':
  #     d['table'] = self.table
  #   if self.scope in ('rows', 'cells'):
  #     d['rows'] = self.rows
  #   if self.scope in ('columns', 'cells'):
  #     # NOTE: Currently a single column only
  #     d['columns'] = self.columns
  #   if self.scope in ('cells'):
  #     d['values'] = self.values.tolist() if isinstance(self.values, pd.Series) else []
  #   return d

  # def as_mask(self, df: pd.DataFrame) -> pd.DataFrame:
  #   assert self.table is not None
  #   mask = pd.DataFrame(
  #     data=False, columns=df.columns, index=df.index, dtype='boolean'
  #   )
  #   rows = self.rows or slice(None)
  #   columns = self.columns or slice(None)
  #   mask.loc[rows, columns] = True
  #   return mask

class Report:

  def __init__(
    self,
    results: Iterable[Result],
    target: Target,
    input: Value = None,
    output: Value = None
  ) -> None:
    self.results: List[Result] = list(results)
    self.target = target
    self.input = input
    self.output = output

  def __repr__(self):
    return f'Report({self.target})'

  def to_df(self, explode: bool = True) -> pd.DataFrame:
    dicts = [result.to_dict() for result in self.results]
    df = pd.DataFrame(dicts)
    if explode:
      df = df.explode('table').explode('column')
      mask = df['value'].isnull() & df['row'].notnull()
      df = pd.concat([
        df[~mask].explode(['row', 'value']),
        df[mask].explode('row')
      ])
      df.sort_index(inplace=True)
    # df = df.convert_dtypes()
    # df = df.mask(df.applymap(lambda x: x is None), '')
    # df = df[df['status'] != 'pass']
    # print(tabulate.tabulate(df, headers='keys', showindex=False, tablefmt='github'))
    return df

# class CheckResults:

#   def __init__(self, results: List[Result]) -> None:
#     self.results = results

#   def as_message_grid(self, table: str, df: pd.DataFrame) -> pd.DataFrame:
#     # TODO: Rewrite with numpy
#     grid = pd.DataFrame(
#       data='', columns=df.columns, index=df.index, dtype='string'
#     )
#     for result in self.results:
#       if result.table == table and result.scope == 'cells' and result.message:
#         mask = result.as_mask(df)
#         grid = grid.mask(mask, grid + f'* {result.message}\n')
#     return grid.mask(grid == '', pd.NA)


# https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html


# ---- Register checks ----

def check(
  fn: Callable = None,
  *,
  args: Dict[str, Scope] = None,
  message: str = None,
  name: str = None,
  severity: Literal['error', 'warning'] = 'error',
  axis: Literal['row', 'column', 'table'] = None,
  requires: Callable = None
) -> Callable:

  def wrapper(fn: Callable):
    # parameters = [Parameter('cls', kind=Parameter.POSITIONAL_OR_KEYWORD)]
    parameters = []
    reserved = []
    for param in inspect.signature(fn).parameters.values():
      if param.kind == Parameter.KEYWORD_ONLY:
        parameters.append(param.replace(kind=Parameter.POSITIONAL_OR_KEYWORD))
        reserved.append(param.name)
    parameters.append(Parameter('kwargs', kind=Parameter.VAR_KEYWORD))
    sig = Signature(parameters, return_annotation=Check)

    # TODO: Augment docstring
    @makefun.with_signature(sig, func_name=fn.__name__, doc=fn.__doc__, check=True)
    def method(**kwargs):
      fn_kwargs = {key: kwargs[key] for key in kwargs if key in reserved}
      cls_kwargs = {
        'args': args,
        'message': message,
        'name': name,
        'severity': severity,
        'axis': axis,
        'requires': filter_kwargs(requires, **fn_kwargs) if requires else None,
        **{key: kwargs[key] for key in kwargs if key not in reserved}
      }
      return Check(fn, kwargs=fn_kwargs, **cls_kwargs)

    # method_name = name or fn.__name__
    # if hasattr(Check, method_name):
    #   raise ValueError(f'Check already has an attribute named {method_name}')
    # setattr(Check, method_name, classmethod(method))
    return method

  if fn is None:
    return wrapper
  return wrapper(fn)


# def check(fn: Callable = None, *, name: str = None, error: str = None) -> Callable:

#   def wrapper(fn: Callable):
#     # TODO: Test fn has compatible parameters
#     parameters = [
#       Parameter('cls', kind=Parameter.POSITIONAL_OR_KEYWORD),
#       *[
#         param for param in inspect.signature(fn).parameters.values()
#         if param.kind == Parameter.KEYWORD_ONLY
#       ],
#       Parameter(
#         'name', kind=Parameter.KEYWORD_ONLY, default=name, annotation=str
#       ),
#       Parameter(
#         'error', kind=Parameter.KEYWORD_ONLY, default=error, annotation=str
#       )
#     ]
#     sig = Signature(parameters, return_annotation=Check)

#     # TODO: Augment docstring
#     @makefun.with_signature(sig, func_name=fn.__name__, doc=fn.__doc__)
#     def method(cls, name: str = None, error: str = None, **kwargs) -> Check:
#       return cls(
#         fn,
#         name=name or fn.__name__,
#         error=error,
#         **kwargs
#       )

#     method_name = name or fn.__name__
#     if hasattr(Check, method_name):
#       raise ValueError(f'Check already has an attribute named {method_name}')
#     setattr(Check, method_name, classmethod(method))
#     return fn

#   if fn is None:
#     return wrapper
#   return wrapper(fn)


# class CheckError(Exception):

#   def __init__(self, message: str) -> None:
#       super().__init__(message)
