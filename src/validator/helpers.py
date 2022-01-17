import inspect
from typing import Any, Callable, Iterable


def filter_kwargs(fn: Callable, /, *args: Any, **kwargs: Any) -> Callable:
  params = inspect.signature(fn).parameters
  has_kwargs = any(
    param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
  )
  if not has_kwargs:
    kwargs = {key: kwargs[key] for key in kwargs if key in params}
  return fn(*args, **kwargs)

def stringify(x: Any) -> str:
  """
  Format object for use in a string.

  Examples
  --------
  >>> stringify(123)
  '123'
  >>> stringify('123')
  "'123'"
  """
  if isinstance(x, str):
    return f"'{x}'"
  return str(x)

def stringify_call(name: str, /, *args: Any, **kwargs: Any) -> str:
  """
  Print a function call.

  Examples
  --------
  >>> stringify_call('fn', 'a', 1, b=2)
  "fn('a', 1, b=2)"
  """
  args = [stringify(arg) for arg in args]
  kwargs = [f'{key}={stringify(value)}' for key, value in kwargs.items()]
  return f"{name}({', '.join(args + kwargs)})"

def sort_partial(values: Iterable, order: Iterable) -> list:
  """
  Sort some list elements, leaving others in place.

  Examples
  --------
  >>> sort_partial(['y', 'z', 'x'], order=['x', 'y', 'z'])
  ['x', 'y', 'z']
  >>> sort_partial(['y', 'z', 'x'], order=['x', 'z'])
  ['y', 'x', 'z']
  >>> sort_partial(['y', 'z', 'x'], order=['x'])
  ['y', 'z', 'x']
  >>> sort_partial(['y', 'z', 'x'], order=['x', 'a', 'y'])
  ['x', 'z', 'y']
  """
  result = list(values)
  order = [x for x in order if x in values]
  position = 0
  for i, value in enumerate(values):
    if value in order:
      result[i] = order[position]
      position += 1
  return result
