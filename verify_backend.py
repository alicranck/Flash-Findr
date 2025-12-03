import sys
import os

# Add backend to path
sys.path.append(os.path.abspath("backend"))

try:
    from app.main import app
    from app.core.session_manager import SessionManager
    print("Backend imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Test Session Manager
manager = SessionManager.get_instance()
assert not manager.is_active(), "Session should be inactive initially"

sid = "test-session"
assert manager.start_session(sid), "Should be able to start session"
assert manager.is_active(), "Session should be active"
assert not manager.start_session("another"), "Should not be able to start another session"

manager.end_session(sid)
assert not manager.is_active(), "Session should be inactive after end"

print("SessionManager logic verified.")
