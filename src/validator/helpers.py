import typing
from typing import Dict, List, Literal

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

