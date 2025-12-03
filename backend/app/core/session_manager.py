import threading
from typing import Optional


class SessionManager:
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.active_session_id: Optional[str] = None
        self._session_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start_session(self, session_id: str) -> bool:
        """
        Attempts to start a new session.
        Returns True if successful, False if a session is already active.
        """
        with self._session_lock:
            if self.active_session_id is not None:
                return False
            self.active_session_id = session_id
            return True

    def end_session(self, session_id: str):
        """
        Ends the current session if the ID matches.
        """
        with self._session_lock:
            if self.active_session_id == session_id:
                self.active_session_id = None

    def is_active(self) -> bool:
        with self._session_lock:
            return self.active_session_id is not None

    def get_active_session(self) -> Optional[str]:
        with self._session_lock:
            return self.active_session_id
