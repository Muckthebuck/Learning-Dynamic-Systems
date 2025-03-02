import sqlite3
import threading
import queue
import signal
import sys
import pickle
import numpy as np
from types import SimpleNamespace
from enum import Enum

class SPSType(Enum):
    OPEN_LOOP = "open loop"
    CLOSED_LOOP = "closed loop"

    def __eq__(self, other):
        if isinstance(other, SPSType):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __str__(self):
        return self.value

class Database:
    """
    Database class to manage storage and retrieval of system states (ss) and data.

    Attributes:
        db_name (str): Name of the SQLite database file.
        lock (threading.Lock): Lock for thread-safe access.
        subscribers (dict): Dictionary of subscribers for pub-sub mechanism.
        queues (dict): Queues to hold messages for subscribers.

    Table Schema:
        ss:
            - id: INTEGER PRIMARY KEY AUTOINCREMENT
            - ss: BLOB (Serialized system state)
            - timestamp: DATETIME (Default: CURRENT_TIMESTAMP)
        
        data:
            - id: INTEGER PRIMARY KEY AUTOINCREMENT
            - data: BLOB (Serialized SimpleNamespace containing y, u, r, sps_type)
            - timestamp: DATETIME (Default: CURRENT_TIMESTAMP)
    """

    def __init__(self, db_name="pipeline.db"):
        self.db_name = db_name
        self.lock = threading.Lock()
        self.subscribers = {"ss": [], "data": []}
        self.queues = {"ss": queue.Queue(), "data": queue.Queue()}
        self._init_db()
        signal.signal(signal.SIGINT, self._shutdown)

    def _init_db(self):
        """Initialize database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ss (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ss BLOB,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data BLOB,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _get_connection(self):
        """Get SQLite connection."""
        return sqlite3.connect(self.db_name, check_same_thread=False)

    def _serialize(self, obj):
        """Serialize Python object to binary using pickle."""
        return pickle.dumps(obj)

    def _deserialize(self, blob):
        """Deserialize binary blob back to Python object and handle enum conversions."""
        obj = pickle.loads(blob)

        # Automatic Enum conversion for SimpleNamespace objects
        if isinstance(obj, SimpleNamespace) and hasattr(obj, 'sps_type'):
            if isinstance(obj.sps_type, str):
                try:
                    obj.sps_type = SPSType(obj.sps_type)
                except KeyError:
                    raise ValueError(f"Invalid sps_type value: {obj.sps_type}")

        return obj


    def write_ss(self, ss):
        """Write system state to database, replacing any existing entry."""
        serialized_ss = self._serialize(ss)
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM ss")
                cursor.execute("INSERT INTO ss (ss) VALUES (?)", (serialized_ss,))
                conn.commit()
                print("[DB] SS written")
                self._notify("ss", ss)

    def read_latest_ss(self):
        """Read the latest system state from the database."""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ss FROM ss ORDER BY timestamp DESC LIMIT 1")
                result = cursor.fetchone()
                return self._deserialize(result[0]) if result else None

    def write_data(self, data):
        """
        Write data to the database, ensuring only one entry exists.

        Args:
            data (SimpleNamespace): Must contain y, u, r, and sps_type.
        """
        if not isinstance(data, SimpleNamespace) or not hasattr(data, 'sps_type'):
            raise ValueError("Data must be a SimpleNamespace with attribute 'sps_type'")

        if not isinstance(data.sps_type, SPSType):
            raise ValueError(f"sps_type must be an instance of SPSType Enum, got {type(data.sps_type)}")

        # Convert Enum to its value for storage
        data.sps_type = data.sps_type.value

        serialized_data = self._serialize(data)
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM data")
                cursor.execute("INSERT INTO data (data) VALUES (?)", (serialized_data,))
                conn.commit()
                print("[DB] Data written")
                self._notify("data", data)

    def read_latest_data(self):
        """Read the latest data entry from the database."""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT data FROM data ORDER BY timestamp DESC LIMIT 1")
                result = cursor.fetchone()
                if result:
                    data = self._deserialize(result[0])
                    data.sps_type = SPSType(data.sps_type)  # Convert back to Enum
                    return data
                return None

    def subscribe(self, topic, callback):
        """Subscribe a callback function to a topic."""
        if topic in self.subscribers:
            thread = threading.Thread(target=self._subscriber_worker, args=(topic, callback), daemon=True)
            self.subscribers[topic].append(thread)
            thread.start()
            print(f"[PubSub] Subscriber added to {topic}")
        else:
            raise ValueError("Invalid topic")

    def _subscriber_worker(self, topic, callback):
        """Worker thread that continuously processes messages from the queue."""
        while True:
            message = self.queues[topic].get()
            callback(message)

    def _notify(self, topic, message):
        """Notify subscribers about new messages on the given topic."""
        if topic in self.queues:
            self.queues[topic].put(message)
            print(f"[PubSub] Notified subscriber of {topic}")

    def _shutdown(self, signum, frame):
        """Gracefully shut down the system on receiving termination signals."""
        print("[DB] Shutting down gracefully...")
        sys.exit(0)
