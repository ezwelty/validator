"""Type aliases."""
from typing import Any, Dict, Hashable, List, Literal, Optional, Protocol, Tuple, Union

import pandas as pd

from .check import Check
from .targets import Target

__all__ = [
    'Axis',
    'ColumnData',
    'CheckFunction',
    'Data',
    'FlatSchemaDict',
    'SchemaDict',
    'TableData',
    'TablesData',
    'ParentData',
    'Valid',
]

ColumnData = pd.Series
TableData = pd.DataFrame
TablesData = Dict[Hashable, pd.DataFrame]
Data = Union[ColumnData, TableData, TablesData]
ParentData = Union[TableData, TablesData]

Axis = Literal['row', 'column', 'table']
Valid = Union[Optional[bool], Dict[Hashable, Optional[bool]], pd.Series]


class CheckFunction(Protocol):
    """Check function."""

    def __call__(
        self, __data: Data, *parents: ParentData, **kwargs: Any
    ) -> Union[Valid, Data, Tuple[Valid, Data]]:
        """Check function call."""
        ...


# Recursive type alias supported by Pylance and maybe other type checkers
SchemaDict = Dict[Target, Union[Check, List[Check], 'SchemaDict']]
FlatSchemaDict = Dict[Target, Check]
