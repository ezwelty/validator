from typing import Callable, Dict, Hashable

from . import column
from . import table
from . import tables
from . import custom
from .. import helpers

__all__ = [
  'column', 'table', 'tables', 'custom', 'register_checks', 'get_check'
]

REGISTRY: Dict[Hashable, Callable] = {}

def register_check(check: Callable, *keys: str) -> None:
  """
  Add a check to the registry.

  The check key needs to be a valid module (dot) path,
  since it is also loaded to the `custom` module.
  """
  key = '.'.join(keys)
  helpers.set_module_path(key, value=check, root=custom)
  REGISTRY[key] = check

def get_check(*keys: str) -> Callable:
  """Retrieve a check by its key."""
  return REGISTRY['.'.join(keys)]

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
