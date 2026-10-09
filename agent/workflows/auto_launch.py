import subprocess
import socket
import time
import os
import signal

def wait_for_port(port, host="localhost", timeout=1800):
    print(f"  Waiting for port {port}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"  Port {port} is ready!")
                return True
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Port {port} not ready after {timeout}s")

print("S2_API_KEY set:", "S2_API_KEY" in os.environ)
S2_API_KEY = os.environ["S2_API_KEY"]

print("[vllm] Starting...")
vllm = subprocess.Popen(
    "CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B "
    "--dtype auto --port 30001 --max-model-len 40960 --host 0.0.0.0",
    shell=True, start_new_session=True,
)
wait_for_port(30001)

print("[mcp] Starting...")
mcp = subprocess.Popen(
    f"export S2_API_KEY={S2_API_KEY} && "
    "python -m dr_agent.mcp_backend.main --port 8000 --host 0.0.0.0",
    shell=True, start_new_session=True,
)
wait_for_port(8000)

print("[query] Running...")
query = subprocess.Popen(
    f"export S2_API_KEY={S2_API_KEY} && "
    "cd /weka/nora-default/varshak/dr-tulu/agent/workflows && python query_direct.py",
    shell=True, start_new_session=True,
)
try:
    query.wait(timeout=60 * 60 * 60)
except subprocess.TimeoutExpired:
    print("[timeout] 18h elapsed.")
finally:
    print("[cleanup] Sending SIGTERM to all subprocesses...")
    for p in (query, mcp, vllm):
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(10)
    print("[cleanup] Sending SIGKILL to any survivors...")
    for p in (query, mcp, vllm):
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
