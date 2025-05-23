"""
Defines the DataConnector interface and implementations for CSV, Excel, and JSON files.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, List
import json

import pandas as pd

class DataConnector(ABC):
    """
    Abstract base class for data connectors.
    All connectors must implement `load()`, and may override `load_in_chunks()`.
    """

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load the entire dataset into a pandas DataFrame.
        """
        pass

    def load_in_chunks(self, chunk_size: int) -> Iterator[pd.DataFrame]:
        """
        Yield the dataset in chunks of size `chunk_size`.
        Default implementation reads all data and yields it once.
        Override for true streaming connectors.
        """
        df = self.load()
        yield df


class CSVConnector(DataConnector):
    """
    Connector for CSV files.
    Uses pandas.read_csv and supports chunked reads.
    """

    def __init__(
        self,
        filepath: str,
        delimiter: str = ",",
        encoding: str = "utf-8",
        parse_dates: Optional[List[str]] = None,
        **read_csv_kwargs
    ):
        self.filepath = filepath
        self.delimiter = delimiter
        self.encoding = encoding
        self.parse_dates = parse_dates
        self.read_csv_kwargs = read_csv_kwargs

    def load(self) -> pd.DataFrame:
        return pd.read_csv(
            self.filepath,
            sep=self.delimiter,
            encoding=self.encoding,
            parse_dates=self.parse_dates,
            **self.read_csv_kwargs
        )

    def load_in_chunks(self, chunk_size: int) -> Iterator[pd.DataFrame]:
        for chunk in pd.read_csv(
            self.filepath,
            sep=self.delimiter,
            encoding=self.encoding,
            parse_dates=self.parse_dates,
            chunksize=chunk_size,
            **self.read_csv_kwargs
        ):
            yield chunk


class ExcelConnector(DataConnector):
    """
    Connector for Excel files.
    Uses pandas.read_excel; concatenates multiple sheets if requested.
    """

    def __init__(
        self,
        filepath: str,
        sheet_name: Optional[str] = 0,
        engine: Optional[str] = None,
        **read_excel_kwargs
    ):
        self.filepath = filepath
        self.sheet_name = sheet_name
        self.engine = engine
        self.read_excel_kwargs = read_excel_kwargs

    def load(self) -> pd.DataFrame:
        df = pd.read_excel(
            self.filepath,
            sheet_name=self.sheet_name,
            engine=self.engine,
            **self.read_excel_kwargs
        )
        if isinstance(df, dict):
            # concatenate all sheets
            return pd.concat(df.values(), ignore_index=True)
        return df


class JSONConnector(DataConnector):
    """
    Connector for JSON files.
    Supports both in-memory JSON arrays and line-delimited JSON.
    Flattens nested structures via pandas.json_normalize.
    """

    def __init__(
        self,
        filepath: str,
        orient: str = "records",
        lines: bool = False,
        **json_kwargs
    ):
        self.filepath = filepath
        self.orient = orient
        self.lines = lines
        self.json_kwargs = json_kwargs

    def load(self) -> pd.DataFrame:
        if self.lines:
            return pd.read_json(
                self.filepath,
                orient=self.orient,
                lines=True,
                **self.json_kwargs
            )
        with open(self.filepath, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return pd.json_normalize(obj, **self.json_kwargs)

