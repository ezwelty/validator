import inspect
import keyword
import types
import typing
from typing import Any, Callable, Dict, List, Literal


Scope = Literal['column', 'table', 'tables']
ARGUMENT_SCOPES: Dict[str, Scope] = {
  's': 'column',
  'column': 'column',
  'df': 'table',
  'table': 'table',
  'dfs': 'tables',
  'tables': 'tables'
}
RANKS: Dict[Scope, int] = {
  key: i for i, key in enumerate(typing.get_args(Scope))
}

# def max_scope(*scopes: Scope) -> Scope:
#   """
#   Examples:
#     >>> max_scope('table', 'tables', 'column')
#     'tables'
#   """
#   rank = max(RANKS[scope] for scope in scopes)
#   return list(RANKS)[rank]

def superscopes(scope: Scope) -> List[str]:
  """
  Examples:
    >>> superscopes('column')
    ['table', 'tables']
    >>> superscopes('table')
    ['tables']
    >>> superscopes('tables')
    []
  """
  return list(RANKS)[RANKS[scope] + 1:]

def is_module_path(path: str) -> bool:
  """Check whether a string is a valid Python module (dot) path."""
  parts = path.split('.')
  for part in parts:
    if not part.isidentifier() or keyword.iskeyword(str):
      return False
  return True

def set_module_path(path: str, value: Any, root: types.ModuleType) -> Any:
  """Set a value on a module path, creating child modules as needed."""
  if not is_module_path(path):
    raise ValueError(f"'{path}' is not a valid module path")
  names = path.split('.')
  # Dry run
  node = root
  i = 0
  for i, name in enumerate(names[:-1]):
    if hasattr(node, name):
      if not isinstance(getattr(node, name), types.ModuleType):
        raise ValueError(
          f"'{node.__name__}.{name}' already exists and is not a module"
        )
      node = getattr(node, name)
    else:
      break
  if (i + 2) == len(names):
    if hasattr(node, names[-1]) and isinstance(getattr(node, names[-1]), types.ModuleType):
        raise ValueError(
          f"'{node.__name__}.{names[-1]}' already exists and is a module"
        )
  # Apply path
  node = root
  for name in names[:-1]:
    if not hasattr(node, name):
      setattr(node, name, types.ModuleType(f'{node.__name__}.{name}'))
    node = getattr(node, name)
  setattr(node, names[-1], value)
  return getattr(node, names[-1])

def filter_kwargs(fn: Callable, *args: Any, **kwargs: Any) -> Callable:
  params = inspect.signature(fn).parameters
  has_kwargs = any(
    param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
  )
  if not has_kwargs:
    kwargs = {key: kwargs[key] for key in kwargs if key in params}
  return fn(*args, **kwargs)
