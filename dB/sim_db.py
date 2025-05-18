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
from multiprocessing import Manager
import logging
import redis
import redis.client
from redis.backoff import ExponentialBackoff
from redis.retry import Retry
from redis.exceptions import (
   BusyLoadingError,
   ConnectionError,
   TimeoutError
)
# def singleton_m(cls):
#     """
#     Singleton decorator to ensure only one instance of the class is created, 
#     even across multiple processes using multiprocessing.Manager.
#     """
#     instances = {}
#     manager = Manager()
    
#     def get_instance(*args, **kwargs):
#         key = (cls, args, frozenset(kwargs.items()))
#         if key not in instances:
#             # Using the Manager to store the instance in shared memory
#             instances[key] = cls(*args, **kwargs, manager=manager)
#         return instances[key]

#     return get_instance

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

class Database:
    """
    Database class to manage storage and retrieval of system states (ss) and data with connection pooling.

    Attributes:
        redis_client (redis.StrictRedis): Redis client for pub-sub mechanism.
        logger (logging.Logger): Logger for logging messages.
        pubsub (redis.client.PubSub): Redis pub-sub object for subscribing to topics.
        db_name (str): Name of the SQLite database file. Default: sim.db
        lock (threading.Lock): Lock for thread-safe access.
        subscribers (dict): Dictionary of subscribers for pub-sub mechanism.
        pool (Queue): Connection pool for SQLite connections.
        POOL_SIZE (int): Size of the connection pool.


    Table Schema:
        archive_ss:
            - id: INTEGER PRIMARY KEY AUTOINCREMENT
            - data: BLOB
            - timestamp: DATETIME (Default: CURRENT_TIMESTAMP)
        
        archive_data:
            - id: INTEGER PRIMARY KEY AUTOINCREMENT
            - data: BLOB
            - timestamp: DATETIME (Default: CURRENT_TIMESTAMP)
        
        archive_controller:
            - id: INTEGER PRIMARY KEY AUTOINCREMENT
            - data: BLOB
            - timestamp: DATETIME (Default: CURRENT_TIMESTAMP)
    """

    POOL_SIZE = 5

    def __init__(self, db_name: str = "sim.db", redis_host: str = "localhost", 
                 redis_port: int = 6379, redis_db: int = 0,
                 logger: Optional[logging.Logger] = None) -> None:
        retry = Retry(ExponentialBackoff(), 3)
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, 
                                              retry=retry, retry_on_error=[BusyLoadingError, ConnectionError, TimeoutError])
        self.logger = logger or logging.getLogger(__name__)
        self.pubsub = self.redis_client.pubsub()
        self.possible_topics = ["ss", "data", "controller"]
        self.db_name = db_name
        self.lock = threading.Lock()

        self.pool = Queue(maxsize=self.POOL_SIZE)
        self._init_pool()
        self._init_db()

        self.pubsub_threads = []
        self.subscriptions = {}

        self.logger.debug(f"[DB] Database initialized with name: {self.db_name}")
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

    def _clear_db(self) -> None:
        """Clear all tables in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS archive_ss")
        cursor.execute("DROP TABLE IF EXISTS archive_data")
        cursor.execute("DROP TABLE IF EXISTS archive_controller")
        conn.commit()
        self._release_connection(conn)

    def _init_db(self) -> None:
        """Initialize database tables if they do not exist."""
        # make sure the database is cleared before creating tables
        self._clear_db()
        # create tables
        self.logger.debug("[DB] Initializing database tables...")
        conn = self._get_connection()
        cursor = conn.cursor()
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
            CREATE TABLE IF NOT EXISTS archive_controller (
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

    def _deserialize(self, blob: dict) -> Any:
        """Deserialize binary blob stored in data field back to Python object and handle enum conversions."""
        obj = pickle.loads(blob["data"])
        if isinstance(obj, SimpleNamespace) and hasattr(obj, 'sps_type'):
            if isinstance(obj.sps_type, str):
                try:
                    obj.sps_type = SPSType(obj.sps_type)
                except KeyError:
                    raise ValueError(f"Invalid sps_type value: {obj.sps_type}")
        return obj

    def write_ss(self, ss: SimpleNamespace) -> None:
        """Write system state to database and archive it."""
        self.write_table("ss", ss)

    def write_data(self, data: SimpleNamespace) -> None:
        """Write data to the database and archive it."""
        if not isinstance(data.sps_type, SPSType):
            raise ValueError(f"sps_type must be an instance of SPSType Enum, got {type(data.sps_type)}")
        self.write_table("data", data)

    def write_controller(self, controller: SimpleNamespace) -> None:
        """Write controller data to the database."""
        self.write_table("controller", controller)

    def write_table(self, table: str, data: Any) -> None:
        """Write data to the specified table."""
        serialized_data = self._serialize(data)
        self.redis_client.publish(table, serialized_data)
        self.logger.debug(f"[DB] Published {table} to Redis")
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(f"INSERT INTO archive_{table} (data) VALUES (?)", (serialized_data,))
            conn.commit()
            self._release_connection(conn)
            self.logger.debug(f"[DB] Data written to {table}")


    def get_latest_controller(self) -> Optional[Any]:
        """Get the latest controller data."""
        controller = self.read_latest("controller")
        if controller:
            return controller
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

    # def _notify(self, topic: str, message: Any) -> None:
    #     """Notify subscribers about new messages on the given topic."""
    #     self.logger.debug(self.subscribers)
    #     for callback in self.subscribers[topic]:
    #         self.logger.debug(callback)
    #         callback(message)
    def _restart_pubsub(self) -> None:
        """Restart Redis pubsub subscriptions after a connection failure."""
        self.logger.debug("[DB] Restarting Redis pubsub...")
        # Stop all existing threads
        for thread in self.pubsub_threads:
            thread.stop()
        self.pubsub_threads.clear()

        # Close existing pubsub object if open
        try:
            self.pubsub.close()
        except Exception as e:
            self.logger.warning(f"[DB] Exception when closing pubsub: {e}")

        # Create new pubsub object
        self.pubsub = self.redis_client.pubsub()

        # Resubscribe to all topics
        for topic, callback in self.subscriptions.items():
            self.pubsub.subscribe(**{topic: callback})

        # Start new pubsub listener thread
        thread = self.pubsub.run_in_thread(
            sleep_time=0.001, daemon=True, exception_handler=self.exception_handler
        )
        self.pubsub_threads.append(thread)
        self.logger.info("[DB] Redis pubsub subscriptions restarted.")

    def exception_handler(self, ex: Exception, pubsub: redis.client.PubSub, thread: threading.Thread) -> None:
        """Handle exceptions in the subscriber thread."""
        self.logger.error(f"Exception in subscriber thread: {ex}")
        self._restart_pubsub()
    
    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """Subscribe a callback function to a topic."""
        self.logger.debug(f"[DB] Subscribing to topic: {topic}")
        if topic not in self.possible_topics:
            raise ValueError(f"Invalid topic: {topic}. Possible topics are: {self.possible_topics}")
        self.pubsub.subscribe(**{topic: callback})
        thread = self.pubsub.run_in_thread(sleep_time=0.001, daemon=True, exception_handler=self.exception_handler)
        self.pubsub_threads.append(thread)
        self.logger.debug(f"[DB] Subscribed to topic: {topic}")

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """Subscribe a callback function to a topic with resiliency."""
        self.logger.debug(f"[DB] Subscribing to topic: {topic}")
        if topic not in self.possible_topics:
            raise ValueError(f"Invalid topic: {topic}. Possible topics are: {self.possible_topics}")

        self.subscriptions[topic] = callback

        # Subscribe on the pubsub object
        self.pubsub.subscribe(**{topic: callback})

        # Start listener thread if not running
        if not self.pubsub_threads:
            self.pubsub_threads = []
        thread = self.pubsub.run_in_thread(
            sleep_time=0.001, daemon=True, exception_handler=self.exception_handler
        )
        self.pubsub_threads.append(thread)
        self.logger.debug("[DB] Started Redis pubsub listener thread.")

        self.logger.debug(f"[DB] Subscribed to topic: {topic}")
    def _shutdown(self, signum: int, frame: Any) -> None:
        """Gracefully shut down the system on receiving termination signals."""
        self.logger.info("[DB] Shutting down gracefully...")
        for thread in self.pubsub_threads:
            thread.stop()
        self.pubsub.close()
        # Close all connections in the pool
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()
        sys.exit(0)

if __name__ == "__main__":
    # monitor the redis server for changes
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    db = Database()
    r = db.redis_client
    with r.monitor() as m:
        for command in m.listen():
            db.logger.debug(f"[DB] Redis command: {command}")