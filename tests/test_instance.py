import socket
import subprocess
import sys
import time
from pathlib import Path

from localsearch.instance import InstanceLock, free_port

ROOT = Path(__file__).resolve().parents[1]


def test_second_lock_on_same_dir_fails(tmp_path):
    a, b = InstanceLock(tmp_path), InstanceLock(tmp_path)
    assert a.acquire()
    assert not b.acquire()
    a.publish("http://127.0.0.1:8740")
    assert b.running_url() == "http://127.0.0.1:8740"
    a.release()
    assert b.acquire()            # released -> next launch owns it
    b.release()


def test_lock_dies_with_the_process(tmp_path):
    code = ("import sys,time; sys.path.insert(0, %r); from localsearch.instance import "
            "InstanceLock as L; l=L(%r); print(l.acquire(), flush=True); time.sleep(30)"
            % (str(ROOT), str(tmp_path)))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "True"
        assert not InstanceLock(tmp_path).acquire()
    finally:
        p.kill()
        p.wait()
    deadline = time.time() + 5
    lk = InstanceLock(tmp_path)
    while not lk.acquire():          # killed process: OS dropped the lock
        assert time.time() < deadline
        time.sleep(0.05)
    lk.release()


def test_free_port_skips_a_busy_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    busy = s.getsockname()[1]
    try:
        assert free_port(busy) != busy
    finally:
        s.close()
