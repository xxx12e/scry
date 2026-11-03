"""Background directory watcher: one debounced scheduler thread feeds the engine."""
import logging
import os
import threading
import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .parser import is_supported

log = logging.getLogger("localsearch.watcher")

_FILE, _TREE = 0, 1


class _Scheduler(threading.Thread):
    """Single thread that fires debounced (path, action) jobs."""

    def __init__(self, engine, debounce_s):
        super().__init__(daemon=True, name="localsearch-watch-scheduler")
        self.engine = engine
        self.debounce_s = debounce_s
        self._cond = threading.Condition()
        self._due = {}
        self._stopped = False

    def schedule(self, path, action):
        with self._cond:
            old = self._due.get(path)
            if old is not None and old[1] == _TREE:
                action = _TREE
            self._due[path] = (time.monotonic() + self.debounce_s, action)
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify()

    def run(self):
        while True:
            with self._cond:
                while not self._stopped and not self._due:
                    self._cond.wait()
                if self._stopped:
                    return
                now = time.monotonic()
                ready = [p for p, (dl, _) in self._due.items() if dl <= now]
                if not ready:
                    soonest = min(dl for dl, _ in self._due.values())
                    self._cond.wait(timeout=max(0.01, soonest - now))
                    continue
                jobs = [(p, self._due.pop(p)[1]) for p in ready]
            for path, action in jobs:
                try:
                    if action == _TREE:
                        self.engine.remove_tree(path, defer=True)
                    elif os.path.exists(path):
                        self.engine.update_file(path, defer=True)
                    else:
                        self.engine.remove_file(path, defer=True)
                except Exception as e:
                    log.warning("watch update failed for %s: %s: %s",
                                path, type(e).__name__, e)


class _Handler(FileSystemEventHandler):
    def __init__(self, scheduler, wanted=None, supported=None):
        self.s = scheduler
        self.wanted = wanted or (lambda p: True)
        self.supported = supported or is_supported

    def _want(self, path):
        """Index-worthy: a supported type AND not under a pruned dir (node_modules,
        AppData, .git, ...). Deletions skip this check -- dropping is harmless."""
        return self.supported(path) and self.wanted(path)

    def on_created(self, e):
        if not e.is_directory and self._want(e.src_path):
            self.s.schedule(e.src_path, _FILE)

    def on_modified(self, e):
        if not e.is_directory and self._want(e.src_path):
            self.s.schedule(e.src_path, _FILE)

    def on_deleted(self, e):
        if self.supported(e.src_path):
            self.s.schedule(e.src_path, _FILE)
        else:
            self.s.schedule(e.src_path, _TREE)

    def on_moved(self, e):
        if self.supported(e.src_path):
            self.s.schedule(e.src_path, _FILE)
        else:
            self.s.schedule(e.src_path, _TREE)
        if e.dest_path and self._want(e.dest_path):
            self.s.schedule(e.dest_path, _FILE)


class Watcher:
    def __init__(self, engine, debounce_s):
        self.scheduler = _Scheduler(engine, debounce_s)
        self._dirs = []
        self.handler = _Handler(
            self.scheduler, lambda p: not engine.in_excluded_dir(p, self._dirs),
            lambda p: is_supported(p) or engine.wants_image(p))
        self.observer = Observer()

    def start(self, dirs):
        self.scheduler.start()
        for d in dirs:
            self.observer.schedule(self.handler, str(d), recursive=True)
            self._dirs.append(str(d))
        self.observer.start()
        log.info("watching %s", self._dirs)

    def stop(self):
        try:
            self.observer.stop()
            self.observer.join(timeout=3)
            self.scheduler.stop()
            log.info("watcher stopped")
        except Exception as e:
            log.warning("watcher stop error: %s", e)
