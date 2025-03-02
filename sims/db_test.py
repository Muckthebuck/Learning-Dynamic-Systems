import numpy as np
import time
from sims.sim_db import Database, SPSType
from types import SimpleNamespace

def test_ss_callback(ss):
    print("[Test] Received SS update:", ss)

def test_data_callback(data):
    print("[Test] Received Data update:", data)

def test_database():
    db = Database()

    # Subscribe to SS and Data updates
    db.subscribe("ss", test_ss_callback)
    db.subscribe("data", test_data_callback)

    # Create sample SS as a dictionary of numpy arrays
    ss = {
        "G": np.random.randn(3, 3),
        "H": np.random.randn(3, 3),
        "F": np.random.randn(3, 3),
        "L": np.random.randn(3, 3)
    }

    # Write SS to database
    print("[Test] Writing SS...")
    db.write_ss(ss)

    # Read latest SS
    read_ss = db.read_latest_ss()
    print("[Test] Read SS:", read_ss)

    # Verify SS Integrity
    assert set(ss.keys()) == set(read_ss.keys()), "SS Keys Mismatch!"
    for key in ss.keys():
        assert np.array_equal(ss[key], read_ss[key]), f"SS Mismatch at {key}!"

    # Create sample Data with SPSType Enum
    data = SimpleNamespace(
        y=np.random.randn(100),
        u=np.random.randn(100),
        r=np.random.randn(100),
        sps_type=SPSType.CLOSED_LOOP
    )

    # Write Data to database
    print("[Test] Writing Data...")
    db.write_data(data)

    # Read latest Data
    read_data = db.read_latest_data()
    print("[Test] Read Data:", read_data)

    # Verify Data Integrity
    assert isinstance(read_data, SimpleNamespace), "Data Type Mismatch!"
    assert np.array_equal(data.y, read_data.y), "Data 'y' Mismatch!"
    assert np.array_equal(data.u, read_data.u), "Data 'u' Mismatch!"
    assert np.array_equal(data.r, read_data.r), "Data 'r' Mismatch!"
    assert data.sps_type == read_data.sps_type, "sps_type Mismatch!"

    print("[Test] All tests passed!")

if __name__ == "__main__":
    test_database()
    # Allow time for subscribers to receive messages
    time.sleep(2)
