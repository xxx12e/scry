"""One running app per index dir.

Two instances on the same index dir would each hold the index in memory and
overwrite each other's saves (and the second one used to crash on the busy port
after it had already started a watcher and a model load). The guard is an
OS-level exclusive lock on `<index_dir>/instance.lock`: the OS drops it when the
process dies, so a crash can never leave a stale lock behind. The running
instance publishes its URL next to the lock; a second launch just opens that.
"""
import json
import os
import socket
import sys
from pathlib import Path


class InstanceLock:
    def __init__(self, index_dir):
        self.dir = Path(index_dir)
        self._fh = None

    def acquire(self):
        """True if this process now owns the index dir."""
        self.dir.mkdir(parents=True, exist_ok=True)
        fh = open(self.dir / "instance.lock", "a+")
        try:
            if sys.platform == "win32":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def publish(self, url):
        try:
            tmp = self.dir / "instance.json.tmp"
            tmp.write_text(json.dumps({"url": url, "pid": os.getpid()}), encoding="utf-8")
            tmp.replace(self.dir / "instance.json")
        except OSError:
            pass

    def running_url(self):
        """URL published by the instance that holds the lock (None if unknown)."""
        try:
            return json.loads((self.dir / "instance.json").read_text(encoding="utf-8"))["url"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def release(self):
        if self._fh is not None:
            try:
                self._fh.close()         # closing the handle releases the lock
            finally:
                self._fh = None


def free_port(preferred, tries=10, host="127.0.0.1"):
    """`preferred` if it can be bound, else the next free one (another program
    sitting on 8731 should not stop the app from starting)."""
    for port in range(preferred, preferred + tries):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, port))
            return port
        except OSError:
            continue
        finally:
            s.close()
    return preferred
