from abc import ABC, abstractmethod
from typing import Any, Hashable, List

from .helpers import Scope, stringify


class Target(ABC):

  @classmethod
  def create(cls, **kwargs) -> 'Target':
    """
    Examples:
      >>> Target.create()
      Tables()
      >>> Target.create(table='A')
      Table('A')
      >>> Target.create(column='a')
      Column('a')
    """
    if 'column' in kwargs:
      return Column(**kwargs)
    if 'table' in kwargs:
      return Table(**kwargs)
    return Tables(**kwargs)

  @property
  @abstractmethod
  def scope(self) -> Scope:
    pass

  @property
  @abstractmethod
  def scopes(self) -> List[Scope]:
    pass

  @abstractmethod
  def includes(self, other: 'Target') -> bool:
    pass

  def __repr__(self) -> str:
    args = []
    for i, (key, value) in enumerate(self.__dict__.items()):
      value = stringify(value)
      if i > 0:
        value = f'{key}={value}'
      args.append(value)
    return f"{self.__class__.__name__}({', '.join(args)})"


class Column(Target):

  def __init__(self, column: Hashable = None, table: Hashable = None) -> None:
    if column is None and table is not None:
      raise ValueError('Column name is required with a table name')
    self.column = column
    self.table = table

  @property
  def scope(self) -> Scope:
    return 'column'

  @property
  def scopes(self) -> List[Scope]:
    scopes = [self.scope]
    if self.column is not None:
      scopes.append('table')
    if self.table is not None:
      scopes.append('tables')
    return scopes

  def includes(self, other: Any) -> bool:
    return isinstance(other, Column) and self.__dict__ == other.__dict__


class Table(Target):

  def __init__(self, table: Hashable = None) -> None:
    self.table = table

  @property
  def scope(self) -> Scope:
    return 'table'

  @property
  def scopes(self) -> List[Scope]:
    scopes = [self.scope]
    if self.table is not None:
      scopes.append('tables')
    return scopes

  def includes(self, other: Any) -> bool:
    return (
      (isinstance(other, Column) and self.table == other.table) or
      (isinstance(other, Table) and self.__dict__ == other.__dict__)
    )


class Tables(Target):

  @property
  def scope(self) -> Scope:
    return 'tables'

  @property
  def scopes(self) -> List[Scope]:
    return [self.scope]

  def includes(self, other: Any) -> bool:
    return (
      (isinstance(other, (Column, Table)) and other.table is not None) or
      (isinstance(other, Tables) and self.__dict__ == other.__dict__)
    )
