from typing import Callable, Dict, Hashable, Sequence

from . import column
from . import table
from . import tables
from . import custom
from .. import helpers

__all__ = [
  'column', 'table', 'tables', 'custom', 'register_checks', 'get_check'
]

REGISTRY: Dict[Hashable, Callable] = {}

def register_check(fn: Callable = None, *, key: str = None) -> Callable:
  """
  Add a check to the registry.

  The check key needs to be a valid module (dot) path,
  since it is also loaded to the `custom` module.
  """
  # HACK: Support key as first positional argument
  if isinstance(fn, str):
    key = fn
    fn = None
  if not key:
    key = fn.__name__

  def decorate(fn):
    helpers.set_module_path(key, value=fn, root=custom)
    REGISTRY[key] = fn
    return fn

  if fn:
    return decorate(fn)
  return decorate

def get_check(key: str) -> Callable:
  """Retrieve a check by its key."""
  return REGISTRY[key]

for name, module in [('column', column), ('table', table), ('tables', table)]:
  REGISTRY.update({
    f'{name}.{k}': v for k, v in getattr(module, '__dict__').items()
    if (
      isinstance(v, Callable) and
      not k.startswith('_') and
      hasattr(v, '__dict__') and
      v.__dict__.get('check')
    )
  })
