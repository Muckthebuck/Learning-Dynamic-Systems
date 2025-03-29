import sqlite3
import threading
import signal
import sys
import pickle
import numpy as np
from queue import Queue
from types import SimpleNamespace
from enum import Enum
from typing import Optional, Callable, Any

def singleton(cls):
    """
    Singleton decorator to ensure only one instance of the class is created.
    """
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        key = (cls, args, frozenset(kwargs.items()))
        with lock:
            if key not in instances:
                instances[key] = cls(*args, **kwargs)
        return instances[key]

    return get_instance

class SPSType(Enum):
    """
    Enum representing system process types.
    """
    OPEN_LOOP = "open loop"
    CLOSED_LOOP = "closed loop"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SPSType):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __str__(self) -> str:
        return self.value

@singleton
class Database:
    """
    Database class to manage storage and retrieval of system states (ss) and data with connection pooling.

    Attributes:
        db_name (str): Name of the SQLite database file. Default: sim.db
        lock (threading.Lock): Lock for thread-safe access.
        subscribers (dict): Dictionary of subscribers for pub-sub mechanism.
        pool (Queue): Connection pool for SQLite connections.

    Table Schema:
        ss:
            - id: INTEGER PRIMARY KEY AUTOINCREMENT
            - data: BLOB (Serialized system state)
            - timestamp: DATETIME (Default: CURRENT_TIMESTAMP)
        
        ctrl:
            - id: INTEGER PRIMARY KEY AUTOINCREMENT
            - data: BLOB (Serialized controller F, L)
            - timestamp: DATETIME (Default: CURRENT_TIMESTAMP)
        
        data:
            - id: INTEGER PRIMARY KEY AUTOINCREMENT
            - data: BLOB (Serialized SimpleNamespace containing y, u, r, sps_type)
            - timestamp: DATETIME (Default: CURRENT_TIMESTAMP)
        
        archive_ss:
            - id: INTEGER PRIMARY KEY AUTOINCREMENT
            - data: BLOB
            - timestamp: DATETIME (Default: CURRENT_TIMESTAMP)
        
        archive_data:
            - id: INTEGER PRIMARY KEY AUTOINCREMENT
            - data: BLOB
            - timestamp: DATETIME (Default: CURRENT_TIMESTAMP)
        
        archive_ctrl:
            - id: INTEGER PRIMARY KEY AUTOINCREMENT
            - data: BLOB
            - timestamp: DATETIME (Default: CURRENT_TIMESTAMP)
    """

    POOL_SIZE = 5

    def __init__(self, db_name: str = "sim.db") -> None:
        self.db_name = db_name
        self.lock = threading.Lock()
        self.subscribers = {"ss": [], "data": [], "ctrl": []}
        self.pool = Queue(maxsize=self.POOL_SIZE)
        self._init_pool()
        self._init_db()
        signal.signal(signal.SIGINT, self._shutdown)

    def _init_pool(self) -> None:
        """Initialize the connection pool."""
        for _ in range(self.POOL_SIZE):
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            self.pool.put(conn)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool."""
        return self.pool.get()

    def _release_connection(self, conn: sqlite3.Connection) -> None:
        """Release a connection back to the pool."""
        self.pool.put(conn)

    def _init_db(self) -> None:
        """Initialize database tables if they do not exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ss (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ctrl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archive_ss (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archive_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archive_ctrl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        self._release_connection(conn)

    def _serialize(self, obj: Any) -> bytes:
        """Serialize Python object to binary using pickle."""
        return pickle.dumps(obj)

    def _deserialize(self, blob: bytes) -> Any:
        """Deserialize binary blob back to Python object and handle enum conversions."""
        obj = pickle.loads(blob)
        if isinstance(obj, SimpleNamespace) and hasattr(obj, 'sps_type'):
            if isinstance(obj.sps_type, str):
                try:
                    obj.sps_type = SPSType(obj.sps_type)
                except KeyError:
                    raise ValueError(f"Invalid sps_type value: {obj.sps_type}")
        return obj

    def write_ss(self, ss: Any) -> None:
        """Write system state to database and archive it."""
        serialized_ss = self._serialize(ss)
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ss")
            cursor.execute("INSERT INTO ss (data) VALUES (?)", (serialized_ss,))
            cursor.execute("INSERT INTO archive_ss (data) VALUES (?)", (serialized_ss,))
            conn.commit()
            self._release_connection(conn)
            print("[DB] ss written and archived")
            self._notify("ss", serialized_ss)

    def write_data(self, data: SimpleNamespace) -> None:
        """Write data to the database and archive it."""
        if not isinstance(data.sps_type, SPSType):
            raise ValueError(f"sps_type must be an instance of SPSType Enum, got {type(data.sps_type)}")
        serialized_data = self._serialize(data)
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO data (data) VALUES (?)", (serialized_data,))
            cursor.execute("INSERT INTO archive_data (data) VALUES (?)", (serialized_data,))
            conn.commit()
            self._release_connection(conn)
            print("[DB] data written and archived")
            self._notify("data", serialized_data)
    
    def write_ctrl(self, ctrl: SimpleNamespace) -> None:
        """Write controller data to the database."""
        serialized_ctrl = self._serialize(ctrl)
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ctrl")
            cursor.execute("INSERT INTO ctrl (data) VALUES (?)", (serialized_ctrl,))
            cursor.execute("INSERT INTO archive_ctrl (data) VALUES (?)", (serialized_ctrl,))
            conn.commit()
            self._release_connection(conn)
            print("[DB] ctrl written")
            self._notify("ctrl", serialized_ctrl)

    def get_latest_ctrl(self) -> Optional[Any]:
        """Get the latest controller data."""
        ctrl = self.read_latest("ctrl")
        if ctrl:
            return ctrl
        return None
    
    def get_latest_data(self) -> Optional[Any]:
        """Get the latest data entry."""
        data = self.read_latest("data")
        if data:
            return data
        return None

    def read_latest(self, table: str) -> Optional[Any]:
        """Read the latest entry from the specified table."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT data FROM {table} ORDER BY timestamp DESC LIMIT 1")
        result = cursor.fetchone()
        self._release_connection(conn)
        if result:
            return self._deserialize(result[0])
        return None

    def _notify(self, topic: str, message: Any) -> None:
        """Notify subscribers about new messages on the given topic."""
        for callback in self.subscribers[topic]:
            callback(message)

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """Subscribe a callback function to a topic."""
        if topic in self.subscribers:
            self.subscribers[topic].append(callback)
            print(f"[PubSub] Subscriber added to {topic}")
        else:
            raise ValueError("Invalid topic")

    def _shutdown(self, signum: int, frame: Any) -> None:
        """Gracefully shut down the system on receiving termination signals."""
        print("[DB] Shutting down gracefully...")
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()
        sys.exit(0)
