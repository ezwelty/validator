from __future__ import annotations
import datetime
import inspect
from inspect import Parameter, Signature
import traceback
from typing import (
  Any, Callable, Dict, Hashable, List, Literal, Optional, Tuple, Type,
  TYPE_CHECKING, Union
)

import makefun
import pandas as pd

from .helpers import filter_kwargs, stringify_call, Timer
from .targets import Column, Table, Tables, Target, classify_data, extract_data
if TYPE_CHECKING:
  from .types import Axis, CheckFunction, Data

# ---- Helpers ----

INPUTS: Dict[str, Type[Target]] = {
  's': Column,
  'column': Column,
  'df': Table,
  'table': Table,
  'dfs': Tables,
  'tables': Tables
}

def _test_inputs(fn: Callable, inputs: Dict[str, Type[Target]] = None) -> Dict[str, Type[Target]]:
  """Test inputs against function signature."""
  # Get positional argument names
  names = [
    param.name for param in inspect.signature(fn).parameters.values()
    if param.kind == param.POSITIONAL_OR_KEYWORD
  ]
  # Require at least one positional argument
  if not names:
    raise ValueError('Function must have at least one positional argument')
  # Map each name to an input type, with preference to user choice
  inputs = inputs or {}
  mapped = {
    name: inputs[name] if name in inputs else INPUTS[name]
    for name in names
    if name in INPUTS or name in inputs
  }
  # All arguments must be assigned a type
  missing = [name for name in names if name not in mapped]
  if missing:
    raise ValueError(
      f"Cannot infer input for function arguments {missing}."
      f" Use a standard name ({list(INPUTS)}) or specify with `inputs=`."
    )
  # All types must be valid inputs and unnamed
  invalid = [
    value for value in mapped.values() if not issubclass(value, Target)
  ]
  if invalid:
    raise ValueError(f"Invalid inputs {invalid}")
  # All secondary types must be parents of first argument's type
  child = mapped[names[0]]
  not_parents = [
    mapped[name] for name in names[1:] if child not in mapped[name].CHILDREN
  ]
  if not_parents:
    raise ValueError(
      f"Inputs {not_parents} are not parents of {child} ('{names[0]}')"
    )
  return mapped

def _test_required(
  inputs: Dict[str, Type[Target]], required: List[Target] = None
) -> List[Union[Column, Table]]:
  """Test required against inputs and function signature."""
  required = required or []
  # Test that required are the children of input parents
  parents = list(inputs.values())
  not_children = []
  for child in required:
    if not any(child in parent() for parent in parents):
      not_children.append(child)
  if not_children:
    raise ValueError(
      f'Required {not_children} not children of {parents}'
    )
  return required

def _test_axis(inputs: Dict[str, Type[Target]], axis: str = None) -> Axis:
  """Test axis against inputs."""
  target = list(inputs.values())[0]
  if not axis:
    return target.AXES[0]
  if axis not in target.AXES:
    raise ValueError(f"Axis '{axis}' cannot be used with {target}")
  return axis

def _test_params(fn: Callable, params: Dict[str, Any] = None) -> Dict[str, Any]:
  """Test params against function signature."""
  params = params or {}
  # Get keyword argument names and whether they are required
  sig_params = inspect.signature(fn).parameters.values()
  kwargs = {
    param.name: param.default is inspect._empty
    for param in sig_params if param.kind == param.KEYWORD_ONLY
  }
  # All required arguments must be present
  missing = [
    name for name, required in kwargs.items() if required and name not in params
  ]
  if missing:
    raise ValueError(f"Missing required keyword arguments {missing}")
  # No extra arguments can be present
  if not any(param.kind == param.VAR_KEYWORD for param in sig_params):
    extra = [name for name in params if name not in kwargs]
    if extra:
      raise ValueError(f"Unexpected keyword arguments {extra}")
  return params

def _generate_method(
  fn,
  inputs: Dict[str, Target] = None,
  required: Union[List[Target], Callable] = None,
  test: bool = True,
  axis: Axis = None
) -> Callable:
  """Generate a Check class method."""
  inputs = _test_inputs(fn=fn, inputs=inputs)
  axis = _test_axis(inputs=inputs, axis=axis)
  if not callable(required):
    required = _test_required(inputs=inputs, required=required)

  def wrapper(method):
    fn_arg_names = [
      param.name
      for param in list(inspect.signature(method).parameters.values())[1:]
      if param.kind == param.POSITIONAL_OR_KEYWORD
    ]
    static = {
      'fn': fn,
      'name': method.__name__,
      'inputs': inputs,
      'test': test,
      'axis': axis
    }
    @makefun.wraps(method, check=True)
    def wrapped(**kwargs):
      cls = kwargs.pop(list(kwargs.keys())[0])
      params = {key: kwargs.pop(key) for key in fn_arg_names}
      if callable(required):
        required_list = filter_kwargs(required, **params)
      else:
        required_list = required
      return cls(**static, required=required_list, params=params, **kwargs)
    return wrapped
  return wrapper

class Check:
  """
  Tests and transforms tabular data.

  Parameters
  ----------
  fn
    A function that tests and/or transforms tabular data.

    The first parameter is the data to process - either a column
    (`Series`), table (`DataFrame`), or tables (`Dict[Any, DataFrame]`).
    Additional parameters may be used to pass parent data (table and/or tables).
    These parameters must either follow the naming convention `s` / `column`,
    `df` / `table`, and `dfs` / `tables`, or be defined in `inputs`.

    Keyword-only parameters (after the `*`) may be used to configure the check.

    Returned test results may either be a single `Optional[bool]` (`True`: pass,
    `False`: fail, `None`: skip) or per child (see `axis`) as `Union[Series,
    Dict[Any, bool]]`. To transform the data, either modify it in place (use
    caution), return the test result and transformation together as a `tuple`,
    or return the transformed data alone with `test=False`.
  inputs
    Mapping of positional parameter names in `fn` to target class `Column`,
    `Table` or `Tables`. Only needed if the names do not follow the naming
    convention (see `fn`).
  required
    Named children of input parent data actually required by `fn`. These can be
    other columns of `Table` (e.g. `[Column('other')]`) or other tables or
    columns of `Tables` (e.g. `[Table('main'), Column('other',
    table='main')]`). This allows the check to be skipped if a parent
    is present but a required child is missing.
    If `Callable`, can accept any of the keyword arguments of `fn` (`params`)
    and must return `List[Target]`.
  test
    Whether the returned value (if not a tuple) is a test result
    or transformed data (see `fn`).
  axis
    Whether a vector test result returned by `fn` is by
    - 'row': default for `Table` and `Column`
    - 'column': alternative for `Table`
    - 'table': default for `Tables`
  params
    Values for the keyword-only parameters of `fn`.
  name
    Name (defaults to the name of `fn`).
  message
    Message reported on test failure.
  tag
    Arbitrary tag.

  Examples
  --------
  >>> import pandas as pd
  >>> import typing
  >>>
  >>> check = Check(lambda s: s.isin({0, 1}))
  >>> check(pd.Series([0, 2])).valid
  0     True
  1    False
  dtype: bool

  >>> def in_set(s: pd.Series, *, values: set) -> pd.Series:
  ...   return s.isin(values)
  >>>
  >>> check = Check(in_set, params={'values': {0, 1}})
  >>> check
  Check.in_set(values={0, 1})
  >>> check.inputs
  {'s': <class 'validator.targets.Column'>}
  >>> check.axis
  'row'
  >>> check(pd.Series([0, 2])).valid
  0     True
  1    False
  dtype: bool

  >>> def in_column(
  ...   s: pd.Series, df: pd.DataFrame, *, name: str
  ... ) -> pd.Series:
  ...   return s.isin(df[name])
  >>> check = Check(in_column, params={'name': 'x'}, required=[Column('x')])
  >>> df = pd.DataFrame({'x': [0, 1], 'y': [0, 2]})
  >>> check(df, target=Column('y')).valid
  0     True
  1    False
  Name: y, dtype: bool

  >>> def in_foreign_column(
  ...   s: pd.Series, dfs: Dict[str, pd.DataFrame], *, table: str, column: str
  ... ) -> pd.Series:
  ...   return s.isin(dfs[table][column])
  >>> check = Check(in_foreign_column, params={'table': 'X', 'column': 'x'})
  >>> dfs = {'X': df}
  >>> check(dfs, target=Column('y', table='X')).valid
  0     True
  1    False
  Name: y, dtype: bool

  >>> def to_numeric(s: pd.Series) -> typing.Tuple[pd.Series, pd.Series]:
  ...   numeric = pd.to_numeric(s, errors='coerce')
  ...   valid = s.isnull() | numeric.notnull()
  ...   return valid, numeric
  >>> result = Check(to_numeric)(pd.Series(['0', '#2']))
  >>> result.valid
  0     True
  1    False
  dtype: bool
  >>> result.output
  0    0.0
  1    NaN
  dtype: float64

  A check may also just return transformed data:

  >>> def upper(s: pd.Series) -> pd.Series:
  ...   return s.str.upper()
  >>> result = Check(upper, test=False)(pd.Series(['a']))
  >>> result.valid is None
  True
  >>> result.output
  0    A
  dtype: object
  """

  def __init__(
    self,
    fn: CheckFunction,
    inputs: Dict[str, Type[Target]] = None,
    required: Union[List[Target], Callable] = None,
    test: bool = True,
    axis: Axis = None,
    params: Dict[str, Any] = None,
    message: str = None,
    name: str = None,
    tag: Any = None
  ) -> None:
    self.fn = fn
    self.inputs = _test_inputs(fn=fn, inputs=inputs)
    self.axis = _test_axis(inputs=self.inputs, axis=axis)
    self.params = _test_params(fn=fn, params=params)
    if callable(required):
      required: List[Target] = filter_kwargs(required, **self.params)
    self.required = _test_required(inputs=self.inputs, required=required)
    self.test = test
    self.message = message
    self.name = name or fn.__name__
    self.tag = tag

  def __repr__(self) -> str:
    return stringify_call(f'Check.{self.name}', **self.params)

  def __eq__(self, other: Any):
    return (
      isinstance(other, type(self)) and
      self.fn is other.fn and
      self.name == other.name and
      self.inputs == other.inputs and
      self.params == other.params and
      self.message == other.message and
      self.tag == other.tag
    )

  # def _call(self, *args: Data, **kwargs: Data) -> Union[Valid, Tuple[Valid, Data]]:
  #   """Call check function directly."""
  #   kwargs = inspect.getcallargs(self.fn, *args, **kwargs, **self.params)
  #   return self.fn(**kwargs)

  def _test_call(
    self, inputs: Dict[Type[Target], Data], target: Target = None
  ) -> Tuple[Target, List[Target], Dict[str, Data]]:
    cls = list(self.inputs.values())[0]
    if target is None:
      target = cls()
    elif not isinstance(target, cls):
      raise ValueError(
        f'Target class does not match check target ({cls})'
      )
    # Compile missing targets
    missing = []
    required = {
      param.name: param.default is inspect._empty
      for param in inspect.signature(self.fn).parameters.values()
    }
    # Ancestors (self.inputs)
    for name, cls in self.inputs.items():
      if required[name] and inputs[cls] is None:
        # Assign name based on target
        missing.append(filter_kwargs(cls, **target.__dict__))
    # Descendants (self.required)
    for xtarget in self.required:
      cls = type(xtarget.ancestor)
      # Ignore if ancestor is also missing
      if inputs[cls] is None:
        continue
      if xtarget.table is not None and xtarget.table not in inputs[Tables]:
        missing.append(xtarget)
      elif isinstance(xtarget, Column):
        if xtarget.table is None:
          if xtarget.column not in inputs[Table]:
            missing.append(xtarget)
        elif xtarget.column not in inputs[Tables][xtarget.table]:
          missing.append(xtarget)
    # Format data for check function
    kwargs = {
      name: inputs[cls]
      for name, cls in self.inputs.items()
      if inputs[cls] is not None
    }
    return target, missing, kwargs

  def __call__(
    self, data: Data = None, name: Target = None, target: Target = None
  ) -> 'Result':
    """
    Check the provided tabular data.

    Calls :attr:`fn` and wraps the result in a :class:`Result`.
    Any error raised by `:attr:`fn` is caught and returned in the result.
    Non-scalar test results are cast to a boolean :class:`pandas.Series`.

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

    Examples
    -------
    >>> df = pd.DataFrame({'id': [0, 1], 'x': [0, 2]})
    >>> check = Check(lambda s: s.isin({0, 1}))
    >>> check(df['x']).valid
    0     True
    1    False
    Name: x, dtype: bool

    >>> check = Check(lambda s, df: s.isin(df['id']))
    >>> check(df)
    Traceback (most recent call last):
      ...
    ValueError: Target class does not match check target ...
    >>> check(df, target=Column('x')).valid
    0     True
    1    False
    Name: x, dtype: bool
    >>> check({'t': df}, target=Column('x', table='t')).valid
    0     True
    1    False
    Name: x, dtype: bool

    >>> dfx = df[['x']]
    >>> result = check(dfx, target=Column('x'))
    >>> result.code
    'error'
    >>> result.error
    KeyError('id')

    >>> check = Check(lambda s, df: s.isin(df['id']), required=[Column('id')])
    >>> result = check(dfx, target=Column('x'))
    >>> result.code
    'skip'
    >>> result.missing
    [Column('id')]
    >>> result = check(df['x'])
    >>> result.missing
    [Table()]

    Raises
    ------
    ValueErorr
      Cannot classify data as Column, Table, or Tables
    ValueError
      Target is not a child of data
    ValueError
      Failed to load data for target from data
    ValueError
      Target class does not match check target
    ValueError
      Test result could not be cast to a boolean pandas.Series
    """
    timer = Timer()
    inputs = extract_data(data, name=name, target=target)
    name = name or classify_data(data)()
    target = target or name
    target, missing, kwargs = self._test_call(inputs, target=target)
    input = inputs[type(target)]
    if missing:
      return Result(
        self, target=target, input=input, missing=missing, time=timer.elapsed
      )
    # Execute function
    try:
      result = self.fn(**kwargs, **self.params)
    except Exception as error:
      error.traceback = traceback.format_exc()
      return Result(
        self, target=target, input=input, error=error, time=timer.elapsed
      )
    if isinstance(result, tuple):
      valid, output = result
    elif self.test:
      valid, output = result, None
    else:
      valid, output = None, result
    # Process test result
    if valid is not None and not isinstance(valid, bool):
      try:
        if isinstance(valid, dict):
          valid = pd.Series(valid, dtype='boolean')
        elif isinstance(valid, pd.Series):
          if not pd.api.types.is_bool_dtype(valid):
            valid = valid.astype('boolean', copy=False)
        else:
          raise TypeError()
      except TypeError:
        raise ValueError(
          f'Test result cannot be cast to a boolean pandas.Series'
        )
    return Result(
      self,
      target=target,
      valid=valid,
      input=input,
      output=output,
      time=timer.elapsed
    )

  def serialize(self) -> Dict[str, Any]:
    """
    Get the dictionary representation of a check.

    Only registered checks (created using a `Check` constructor) are supported.
    Result includes :attr:`name`, :attr:`params` if non-empty, and
    :attr:`message` and :attr:`tag` if these differ from the defaults.
    """
    if not hasattr(self.__class__, self.name):
      raise ValueError(f'Check {self.name} is not registered')
    method = getattr(self.__class__, self.name)
    defaults = {
      param.name: param.default
      for param in inspect.signature(method).parameters.values()
      if param.kind == param.KEYWORD_ONLY and param.default is not inspect._empty
    }
    result = {'name': self.name}
    if self.params:
      result['params'] = self.params
    for attr in ['message', 'tag']:
      value = getattr(self, attr)
      if (
        value is not None and (attr not in defaults or value != defaults[attr])
      ):
        result[attr] = value
    return result

  @classmethod
  def deserialize(cls, name: str, params: dict = None, **kwargs: Any) -> 'Check':
    """
    Load a check from a dictionary.

    Only registered checks (created using a `Check` constructor) are supported.

    Parameters
    ----------
    name
      Check name. Must be the name of a Check class method.
    params
      Check parameters (values for the keyword-only parameters of :attr:`fn`).
    **kwargs
      Additional arguments passed to `Check.{name}`.
    """
    params = params or {}
    if not hasattr(cls, name):
      raise ValueError(f'Check {name} is not registered')
    return getattr(cls, name)(**params, **kwargs)


class Result:
  """
  Describes the result of a :class:`Check`.

  Parameters
  ----------
  check
    Check that was called.
  target
    Target of the check.
  input
    Input data for `target`.
  output
    Output (potentially transformed) data for `target`.
  valid
    Validation result.
  error
    Error raised by `check.fn`.
    The original traceback is available from `error.traceback`.
  missing
    Required data found to be missing.
  time
    Execution time.
  """

  def __init__(
    self,
    check: Check,
    target: Target,
    input: Data = None,
    output: Data = None,
    valid: Union[bool, pd.Series] = None,
    error: Exception = None,
    missing: List[Target] = None,
    time: datetime.timedelta = None
  ) -> None:
    self.check = check
    self.target = target
    self.valid = valid
    self.input = input
    self.output = output
    self.error = error
    self.missing = missing
    self.time = time

  def __repr__(self):
    return stringify_call(
      self.__class__.__name__,
      self.code,
      target=self.target,
      check=self.check
    )

  @property
  def code(self) -> Literal['pass', 'fail', 'error', 'skip']:
    """Result code."""
    if self.error is not None:
      return 'error'
    if (
      self.missing is not None or
      (self.valid is None and self.output is None)
    ):
      return 'skip'
    if (
      self.valid is True or
      (isinstance(self.valid, pd.Series) and self.valid.all()) or
      # Only transformed input (no test result)
      (self.valid is None and self.output is not None)
    ):
      return 'pass'
    if (
      self.valid is False or
      (isinstance(self.valid, pd.Series) and not self.valid.all())
    ):
      return 'fail'
    assert False, 'Could not determine result code'

  @property
  def message(self) -> Optional[str]:
    """
    Human-readable message describing the problem (if any).

    The content depends on the :attr:`code`:
    - skip and missing: Which required data were missing.
    - error: The message of the error raised by :attr:`check`.`fn`.
    - fail: :attr:`check`.`message`, formatted based on :attr:`check`.`params`.
    """
    code = self.code
    if code == 'fail' and self.check.message is not None:
      try:
        return eval(f"f'{self.check.message}'", None, self.check.params)
      except SyntaxError:
        # In case unescaped single quotes are present in message template
        return eval(f'f"{self.check.message}"', None, self.check.params)
    if code == 'error':
      return str(self.error)
    if code == 'skip' and self.missing:
      return f'Missing required inputs {self.missing}'

  @property
  def axis(self) -> Optional[Literal['row', 'column', 'table']]:
    """Test result axis, if :attr:`valid` is not a scalar."""
    if self.valid is None or isinstance(self.valid, bool):
      return None
    return self.check.axis

  @property
  def table(self) -> Optional[Union[Hashable, List[Hashable]]]:
    """Name of invalid table(s)."""
    return (
      (self.axis == 'table' and list(self.valid[~self.valid].index)) or
      (isinstance(self.target, (Table, Column)) and self.target.table) or
      None
    )

  @property
  def column(self) -> Optional[Union[Hashable, List[Hashable]]]:
    """Name of invalid column(s)."""
    return (
      (self.axis == 'column' and list(self.valid[~self.valid].index)) or
      (isinstance(self.target, Column) and self.target.column) or
      None
    )

  @property
  def row(self) -> Optional[List[Hashable]]:
    """Name of invalid rows."""
    return (
      (self.axis == 'row' and list(self.valid[~self.valid].index)) or
      None
    )

  @property
  def value(self) -> Optional[List[Hashable]]:
    """Invalid values."""
    return (
      (
        self.axis == 'row' and
        isinstance(self.target, Column) and
        list(self.input[self.valid[~self.valid].index])
      ) or
      None
    )

  def _to_dict(self) -> dict:
    """Get a dictionary representation of the result."""
    return {
      'code': self.code,
      'table': self.table,
      'column': self.column,
      'row': self.row,
      'value': self.value,
      'check': self.check,
      'tag': self.check.tag,
      'message': self.message
    }


def register_check(
  fn: Callable = None,
  *,
  name: str = None,
  inputs: Dict[str, Target] = None,
  required: Union[List[Target], Callable] = None,
  test: bool = True,
  axis: Axis = None,
  message: str = None,
  tag: Any = None
) -> Callable:
  """
  Register a function as a :class:`Check` method.

  Parameters
  ----------
  fn
    Check function (see :attr:`Check.fn`).
  name
    Name of the created :class:`Check` method. Defaults to the name of `fn`.
    `Check.{name}` cannot already exist as a built-in class attribute.
  inputs
    Mapping of positional parameter names to `Column`, `Table`, or `Tables`
    (see :attr:`Check.inputs`).
  required
    Named children of input parents actually used by `fn`
    (see :attr:`Check.required`).
    If `Callable`, can accept any of the keyword arguments of `fn`
    and must return `List[Target]`.
  test
    Whether the return value of `fn`, if not a tuple, is a test result
    or transformed data (see :attr:`Check.test`).
  axis
    Whether a vector test result returned by `fn` is by
    'row', 'column', or 'table' (see :attr:`Check.axis`).
  message
    Default message reported on test failure (see :attr:`Check.message`).
  tag
    Default tag (see :attr:`Check.tag`).

  Returns
  -------
  The function `fn` without modifications.
  """

  def factory(fn: Callable) -> Callable:
    method_name = name or fn.__name__
    if hasattr(Check, method_name):
      raise ValueError(
        f'{Check.__name__}.{method_name} already exists'
      )

    # Build parameters of constructor
    parameters = []
    # reserved = []
    # Add cls as first argument
    parameters.append(Parameter('cls', kind=Parameter.POSITIONAL_OR_KEYWORD))
    # Add keyword-only function arguments as positional-or-keyword arguments
    for param in inspect.signature(fn).parameters.values():
      if param.kind == Parameter.KEYWORD_ONLY:
        parameters.append(param.replace(kind=Parameter.POSITIONAL_OR_KEYWORD))
        # reserved.append(param.name)
    # Add customizable Check parameters as keyword-only arguments
    for key, dtype, default in [('message', str, message), ('tag', Any, tag)]:
      parameters.append(
        Parameter(
          key, default=default, annotation=dtype, kind=Parameter.KEYWORD_ONLY
        )
      )

    @_generate_method(
      fn, inputs=inputs, required=required, test=test, axis=axis
    )
    @makefun.with_signature(
      # Use string 'Check' to match built-in class methods
      Signature(parameters, return_annotation=Check.__name__),
      func_name=method_name,
      qualname=f'{Check.__name__}.{method_name}',
      module_name=Check.__module__,
      doc=None,
      builtin=False
    )
    def constructor(cls, **kwargs: Any) -> Check:
      # fn_kwargs = {key: kwargs.pop(key) for key in reserved}
      # kwargs = {**default, **kwargs}
      # return cls(fn=fn, kwargs=fn_kwargs, **static, **kwargs)
      pass

    setattr(Check, method_name, classmethod(constructor))
    setattr(Check, method_name, getattr(Check, method_name).__get__(Check))
    return fn

  if fn is None:
    return factory
  return factory(fn)
