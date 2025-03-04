import unittest
import sqlite3
import pickle
import numpy as np
from types import SimpleNamespace
from sims.sim_db import Database, SPSType
from typing import Optional

class TestDatabase(unittest.TestCase):
    """
    Unit tests for the Database class, verifying correct functionality
    for writing, reading, archiving, and subscribing to data.
    """

    def setUp(self) -> None:
        """Set up the test database and clear existing tables."""
        self.db = Database("test_pipeline.db")
        self.clear_tables()
        self.check_valid_tables()

    def check_valid_tables(self) -> None:
        """Verify that the expected tables exist in the database."""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            self.db._release_connection(conn)
        expected_tables = {"ss", "data", "archive_ss", "archive_data"}
        self.assertSetEqual(tables & expected_tables, expected_tables)
        unexpected_tables = tables - expected_tables - {"sqlite_sequence"}
        self.assertFalse(unexpected_tables, f"Unexpected tables found: {unexpected_tables}")

    def clear_tables(self) -> None:
        """Delete all rows from the database tables."""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            for table in ["ss", "data", "archive_ss", "archive_data"]:
                cursor.execute(f"DELETE FROM {table}")
            conn.commit()
            self.db._release_connection(conn)

    def test_write_ss(self) -> None:
        """Test writing and reading state-space data."""
        ss = np.array([1.0, 2.0, 3.0])
        self.db.write_ss(ss)
        result = self.db.read_latest("ss")
        np.testing.assert_array_equal(result, ss)

    def test_write_data(self) -> None:
        """Test writing and reading control data."""
        data = SimpleNamespace(y=np.array([1.0]), u=np.array([2.0]), r=np.array([3.0]), sps_type=SPSType.OPEN_LOOP)
        self.db.write_data(data)
        result = self.db.read_latest("data")
        self.assertEqual(result.sps_type, SPSType.OPEN_LOOP)
        np.testing.assert_array_equal(result.y, data.y)
        np.testing.assert_array_equal(result.u, data.u)
        np.testing.assert_array_equal(result.r, data.r)

    def test_invalid_sps_type(self) -> None:
        """Test handling of invalid SPS type."""
        data = SimpleNamespace(y=np.array([1.0]), u=np.array([2.0]), r=np.array([3.0]), sps_type="invalid")
        with self.assertRaises(ValueError):
            self.db.write_data(data)

    def test_read_latest_empty(self) -> None:
        """Test reading the latest data from an empty table."""
        result: Optional[np.ndarray] = self.db.read_latest("ss")
        self.assertIsNone(result)

    def test_archive_ss(self) -> None:
        """Test archiving of state-space data."""
        ss = np.array([4.0, 5.0, 6.0])
        self.db.write_ss(ss)
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM archive_ss ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            self.db._release_connection(conn)
        self.assertIsNotNone(row)
        archived_ss = pickle.loads(row[0])
        np.testing.assert_array_equal(archived_ss, ss)

    def test_archive_data(self) -> None:
        """Test archiving of control data."""
        data = SimpleNamespace(y=np.array([7.0]), u=np.array([8.0]), r=np.array([9.0]), sps_type=SPSType.CLOSED_LOOP)
        self.db.write_data(data)
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM archive_data ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            self.db._release_connection(conn)
        self.assertIsNotNone(row)
        archived_data = pickle.loads(row[0])
        self.assertEqual(archived_data.sps_type, SPSType.CLOSED_LOOP)
        np.testing.assert_array_equal(archived_data.y, data.y)
        np.testing.assert_array_equal(archived_data.u, data.u)
        np.testing.assert_array_equal(archived_data.r, data.r)

    def test_subscribe_notify(self) -> None:
        """Test subscription notifications for state-space data."""
        def callback(msg: bytes) -> None:
            self.received_msg = msg

        self.received_msg: Optional[bytes] = None
        self.db.subscribe("ss", callback)
        ss = np.array([10.0, 11.0, 12.0])
        self.db.write_ss(ss)
        self.assertIsNotNone(self.received_msg)
        np.testing.assert_array_equal(self.db._deserialize(self.received_msg), ss)

if __name__ == "__main__":
    unittest.main()