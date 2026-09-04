import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "nt", "Windows console lifecycle test")
class ServerLifecycleTests(unittest.TestCase):
    def test_console_break_stops_server_and_releases_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        with tempfile.TemporaryDirectory() as data_directory:
            environment = os.environ.copy()
            environment["RVC_STREAMING_PORT"] = str(port)
            environment["RVC_DATA_DIR"] = data_directory
            process = subprocess.Popen(
                [sys.executable, "server.py"],
                cwd=SERVER_DIR,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        output = process.communicate(timeout=2)[0]
                        self.fail(f"server exited before listening:\n{output}")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                            break
                    except OSError:
                        time.sleep(0.05)
                else:
                    self.fail("server did not start listening in time")

                process.send_signal(signal.CTRL_BREAK_EVENT)
                exit_code = process.wait(timeout=10)
                output = process.communicate(timeout=2)[0]

                self.assertEqual(exit_code, 0, output)
                self.assertIn("stopping RVC Server", output)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rebound:
                    rebound.bind(("127.0.0.1", port))
            finally:
                if process.poll() is None:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )


if __name__ == "__main__":
    unittest.main()
