"""Single-instance guard for iRaceEngineer.

Holds an exclusive OS-level lock on a lock file in the temp directory for
the lifetime of the process. If another instance already holds the lock,
acquire() fails and the caller should exit. The lock is released
automatically by the OS if the process exits or crashes, so there is no
stale-lock-file problem.
"""

import logging
import os
import sys
import tempfile
from typing import IO

logger = logging.getLogger(__name__)

LOCK_FILENAME = "iraceengineer.lock"


class SingleInstanceGuard:
    """Exclusive lock preventing two iRaceEngineer instances at once.

    Uses msvcrt.locking() on Windows and fcntl.flock() on POSIX — both are
    advisory byte-range locks tied to the open file handle, so the lock
    disappears when the process exits for any reason (including crash).
    """

    def __init__(self, name: str = LOCK_FILENAME):
        self._path = os.path.join(tempfile.gettempdir(), name)
        self._fh: IO[str] | None = None

    def acquire(self) -> bool:
        """Try to take the lock. Returns True if this is the only instance."""
        self._fh = open(self._path, "w")
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._fh.close()
            self._fh = None
            return False

        # Record the PID for diagnostics (lock already held, so no race)
        try:
            self._fh.seek(0)
            self._fh.write(str(os.getpid()))
            self._fh.flush()
        except OSError:
            pass  # purely informational — lock is what matters

        logger.debug(
            "Acquired single-instance lock at %s (pid %d)", self._path, os.getpid()
        )
        return True

    def release(self):
        """Release the lock and remove the lock file."""
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            os.unlink(self._path)
        except OSError:
            pass  # process is likely exiting — OS would clean up anyway
        finally:
            self._fh = None
