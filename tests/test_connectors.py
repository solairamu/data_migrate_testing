import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pytest
from pathlib import Path
import pandas as pd
from connectors.connectors import CSVConnector, ExcelConnector, JSONConnector

TEST_DATA_DIR = Path(__file__).parent / "test_data"

def test_addresses_csv():
    file_path = TEST_DATA_DIR / 'addresses.csv'
    conn = CSVConnector(file_path)
    df = conn.load()

    # The example CSV has 5 data rows and 6 columns
    assert df.shape == (5, 6)
    # test chunked loading sums to same row count
    chunks = list(conn.load_in_chunks(chunk_size=2))
    total_rows = sum(chunk.shape[0] for chunk in chunks)
    assert total_rows == df.shape[0]


def test_excel_connector_example_file():
    file_path = TEST_DATA_DIR / 'file_example_XLSX_10.xlsx'
    conn = ExcelConnector(file_path)
    df = conn.load()

    # Now 9 rows × 7 real columns (unnamed index dropped)
    assert df.shape == (9, 7)

    for col in ["First Name", "Last Name", "Gender", "Country", "Age", "Date", "Id"]:
        assert col in df.columns


def test_json_connector_simple():
    file_path = TEST_DATA_DIR / 'example_1.json'
    conn = JSONConnector(file_path)
    df = conn.load()

    # JSON object becomes a single-row DataFrame with 3 columns
    assert df.shape == (1, 3)
    assert set(df.columns) == {'fruit', 'size', 'color'}
    # verify values
    assert df.at[0, 'fruit'] == 'Apple'
    assert df.at[0, 'size'] == 'Large'
    assert df.at[0, 'color'] == 'Red'


def test_jsonl_connector_simple():
    file_path = TEST_DATA_DIR / 'example_1.jsonl'
    conn = JSONConnector(file_path, lines=True)
    df = conn.load()

    # JSONL file should load same as JSON
    assert df.shape == (1, 3)
    assert set(df.columns) == {'fruit', 'size', 'color'}
    assert df.iloc[0]['color'] == 'Red'


if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])

