from typing import (
  Any, Dict, Hashable, Union, List, Literal, Optional, Protocol, Tuple
)

import pandas as pd

from .targets import Target
from .check import Check

__all__ = [
  'Axis',
  'ColumnData',
  'CheckFunction',
  'Data',
  'FlatSchemaDict'
  'SchemaDict',
  'TableData',
  'TablesData',
  'ParentData',
  'Valid'
]

ColumnData = pd.Series
TableData = pd.DataFrame
TablesData = Dict[Hashable, pd.DataFrame]
Data = Union[ColumnData, TableData, TablesData]
ParentData = Union[TableData, TablesData]

Axis = Literal['row', 'column', 'table']
Valid = Union[Optional[bool], Dict[Hashable, Optional[bool]], pd.Series]

class CheckFunction(Protocol):
  def __call__(
    self, __data: Data, *parents: ParentData, **kwargs: Any
  ) -> Union[Valid, Data, Tuple[Valid, Data]]: ...

# Recursive type alias supported by Pylance and maybe other type checkers
SchemaDict = Dict[Target, Union[Check, List[Check], 'SchemaDict']]
FlatSchemaDict = Dict[Target, Check]
