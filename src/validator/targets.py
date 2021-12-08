from typing import Any, Hashable, List, Union

from .helpers import Scope


class Column:

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

  def __repr__(self) -> str:
    args = ''
    if self.column is not None:
      args += f'{self.column}'
    if self.table is not None:
      args += f', table={self.table}'
    return f'Column({args})'

  def includes(self, other: Any) -> bool:
    return isinstance(other, Column) and self.__dict__ == other.__dict__


class Table:

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

  def __repr__(self) -> str:
    args = '' if self.table is None else self.table
    return f'Table({args})'

  def includes(self, other: Any) -> bool:
    return (
      (isinstance(other, Column) and self.table == other.table) or
      (isinstance(other, Table) and self.__dict__ == other.__dict__)
    )


class Tables:

  def __init__(self) -> None:
    pass

  @property
  def scope(self) -> Scope:
    return 'tables'

  @property
  def scopes(self) -> List[Scope]:
    return [self.scope]

  def __repr__(self) -> str:
    return 'Tables()'

  def includes(self, other: Any) -> bool:
    return (
      (isinstance(other, (Column, Table)) and other.table is not None) or
      (isinstance(other, Tables) and self.__dict__ == other.__dict__)
    )

Target = Union[Column, Table, Tables]
