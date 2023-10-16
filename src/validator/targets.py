from __future__ import annotations

from abc import ABC
from multiprocessing.sharedctypes import Value
from typing import TYPE_CHECKING, Any, Dict, Hashable, List, Optional, Set, Type

import pandas as pd

from .helpers import stringify_call

if TYPE_CHECKING:
    from .types import Axis, Data


class Target(ABC):
    AXES: List[Axis] = []
    CHILDREN: Set[Type['Target']] = {}

    @classmethod
    def create(cls, **kwargs) -> 'Target':
        """
        Examples
        --------
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
    def parent(self) -> None:
        return None

    @property
    def ancestor(self) -> Optional['Target']:
        node = self
        while node.parent is not None:
            node = node.parent
        if node is self:
            return None
        return node

    @property
    def named(self) -> bool:
        return any(value is not None for value in self.__dict__.values())

    def equals(self, other: Any) -> bool:
        return self.__class__ is other.__class__ and self.__dict__ == other.__dict__

    def matches(self, other: Any) -> bool:
        return (
            type(self) is type(other)
            and list(other.__dict__) == list(self.__dict__)
            and all(self.__dict__[k] in {None, v} for k, v in other.__dict__.items())
        )

    def __contains__(self, other: Any) -> bool:
        return False

    def __repr__(self) -> str:
        filtered = [(k, v) for k, v in self.__dict__.items() if v is not None]
        args = [v for _, v in filtered[:1]]
        kwargs = dict(filtered[1:])
        return stringify_call(self.__class__.__name__, *args, **kwargs)

    def copy(self) -> 'Target':
        return type(self)(**self.__dict__)


class Column(Target):
    AXES: List[Axis] = ['row']
    CHILDREN: Set[Type[Target]] = {}

    def __init__(self, column: Hashable = None, table: Hashable = None) -> None:
        if column is None and table is not None:
            raise ValueError('Column name is required with a table name')
        self.column = column
        self.table = table

    @property
    def parent(self) -> Optional['Table']:
        if self.column is not None:
            return Table(self.table)
        return None


class Table(Target):
    AXES: List[Axis] = ['row', 'column']
    CHILDREN: Set[Type[Target]] = {Column}

    def __init__(self, table: Hashable = None) -> None:
        self.table = table

    @property
    def parent(self) -> Optional['Tables']:
        if self.table is not None:
            return Tables()
        return None

    def __contains__(self, other: Any) -> bool:
        return isinstance(other, Column) and (
            other.table is None or other.table == self.table
        )


class Tables(Target):
    AXES: List[Axis] = ['table']
    CHILDREN: Set[Type[Target]] = {Column, Table}

    def __contains__(self, other: Any) -> bool:
        return isinstance(other, (Column, Table))


def classify_data(data: Data) -> Type[Target]:
    """
    Classify tabular type of input data.

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> classify_data(pd.Series([0, 1]))
    <class 'validator.targets.Column'>
    >>> classify_data(pd.DataFrame({'x': [0, 1]}))
    <class 'validator.targets.Table'>
    >>> classify_data({})
    <class 'validator.targets.Tables'>
    >>> classify_data({'x': pd.DataFrame({'x': [0, 1]})})
    <class 'validator.targets.Tables'>
    """
    if isinstance(data, pd.Series):
        return Column
    if isinstance(data, pd.DataFrame):
        return Table
    if isinstance(data, dict):
        if all(isinstance(value, pd.DataFrame) for value in data.values()):
            return Tables
    raise ValueError(f'Cannot classify data ({type(data)}) as Column, Table, or Tables')


def extract_data(
    data: Data, name: Target = None, target: Target = None
) -> Dict[Type[Target], Data]:
    """
    Extract data.

    Examples
    --------
    >>> df = pd.DataFrame({'x': [0]})
    >>> inputs = extract_data(df)
    >>> inputs == {Tables: None, Table: df, Column: None}
    True
    >>> inputs = extract_data(df, target=Column('x'))
    >>> inputs == {Tables: None, Table: df, Column: df['x']}
    True
    >>> dfs = {'X': df}
    >>> inputs = extract_data(dfs, target=Column('x', table='X'))
    >>> inputs == {Tables: dfs, Table: dfs['X'], Column: dfs['X']['x']}
    True
    """
    cls = classify_data(data)
    # Identify data type
    if name is None:
        name = cls()
    elif not isinstance(name, cls):
        raise ValueError(f'Data recognized as {cls}, not {type(name)}')
    # Check child is actually a child
    if target is not None:
        if (
            isinstance(target, Column)
            and isinstance(name, Table)
            and target.table is None
        ):
            target.table = name.table
        if not target.equals(name) and target not in name:
            raise ValueError(f'{target} is not a child of {name}')
    if target is None:
        target = name
    # Check that table and column names are not None
    none_as_name = False
    if isinstance(name, (Table, Tables)):
        none_as_name |= None in data
    if isinstance(name, Tables):
        none_as_name |= any(None in value for value in data.values())
    if none_as_name:
        raise ValueError(f'Tables and columns cannot be named `None`')
    # Check that table column names are unique
    if isinstance(name, Table):
        seen = set()
        duplicates = [x for x in data if x in seen or seen.add(x)]
        if duplicates:
            raise ValueError(f'Table contains duplicate column names: {duplicates}')
    elif isinstance(name, Tables):
        all_duplicates = {}
        for key, df in data.items():
            seen = set()
            duplicates = [x for x in df if x in seen or seen.add(x)]
            if duplicates:
                all_duplicates[key] = duplicates
        if all_duplicates:
            raise ValueError(
                f'Table(s) contain duplicate column names: {all_duplicates}'
            )
    # Load child data
    inputs = {Tables: None, Table: None, Column: None}
    inputs[type(name)] = data
    if (
        inputs[Tables] is not None
        and inputs[Table] is None
        and isinstance(target, (Column, Table))
        and target.table is not None
        and target.table in inputs[Tables]
    ):
        inputs[Table] = inputs[Tables][target.table]
    if (
        inputs[Table] is not None
        and inputs[Column] is None
        and isinstance(target, Column)
        and target.column is not None
        and target.column in inputs[Table]
    ):
        inputs[Column] = inputs[Table][target.column]
    if inputs[type(target)] is None:
        raise ValueError(f'Failed to extract data for {target} from data')
    return inputs
