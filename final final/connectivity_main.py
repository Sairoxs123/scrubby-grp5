from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager
import socket
import threading
import queue
import math
import time

# ============================================================
#  ESP32 TCP CONFIG  —  replace with your ESP32's IP address
# ============================================================
ESP32_IP   = "192.168.X.X"   # <-- replace with your ESP32's IP
ESP32_PORT = 8001

# ============================================================
#  GLOBALS
# ============================================================
tcp_socket = None
socket_lock = threading.Lock()

robot_state  = "IDLE"
current_patch = None
state_lock   = threading.Lock()

patch_queue  = queue.Queue()

_robot_position = {"x": 0, "y": 0, "angle": 0, "t": 0}
position_lock   = threading.Lock()

serial_response_queue = queue.Queue()
cv_connected          = threading.Event()

SEND_TIMEOUT      = 5
MAX_PATCH_RETRIES = 3
MAX_QUEUE_SIZE    = 10

# ============================================================
#  TCP CONNECTION
# ============================================================
def connect_esp32():
    """Open a TCP connection to the ESP32 and perform handshake."""
    try:
        print(f"[TCP] Connecting to ESP32 at {ESP32_IP}:{ESP32_PORT}...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((ESP32_IP, ESP32_PORT))
        s.settimeout(None)                     # blocking reads from here on
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Handshake
        s.sendall(b"HELLO_ROBOT\n")
        response = _read_line_from_socket(s, timeout=5)
        print(f"[DEBUG] Handshake response: '{response}'")

        if response == "ROBOT_READY":
            print("[OK] ESP32 connected via WiFi TCP.")
            return s

        print("[ERROR] Invalid handshake — expected 'ROBOT_READY'")
        s.close()

    except Exception as e:
        print(f"[ERROR] TCP connection failed: {e}")

    return None


def _read_line_from_socket(s, timeout=None):
    """Read bytes from socket until newline, with optional timeout."""
    buf  = b""
    deadline = (time.time() + timeout) if timeout else None
    s.settimeout(0.05)
    try:
        while True:
            if deadline and time.time() > deadline:
                return ""
            try:
                chunk = s.recv(1)
                if not chunk:
                    return ""
                if chunk == b"\n":
                    return buf.decode(errors="replace").strip()
                buf += chunk
            except socket.timeout:
                continue
    finally:
        s.settimeout(None)


def try_reconnect_esp32():
    global tcp_socket
    delay = 2
    while True:
        print(f"[RECONNECT] Retrying in {delay}s...")
        time.sleep(delay)
        s = connect_esp32()
        if s:
            with socket_lock:
                tcp_socket = s
            with state_lock:
                global robot_state
                robot_state = "IDLE"
            print("[RECONNECT] ESP32 reconnected.")
            return
        delay = min(30, delay * 2)

# ============================================================
#  LIFESPAN
# ============================================================
@asynccontextmanager
async def lifespan(app):
    global tcp_socket, robot_state
    tcp_socket  = connect_esp32()
    robot_state = "IDLE" if tcp_socket else "DISCONNECTED"
    if not tcp_socket:
        print("[FAIL] ESP32 not reachable at startup — scheduler will retry.")
    yield

app = FastAPI(lifespan=lifespan)

# ============================================================
#  POSITION HELPERS
# ============================================================
def get_robot_position():
    with position_lock:
        return _robot_position.copy()

def is_position_fresh(timeout=0.5):
    return time.time() - get_robot_position()["t"] < timeout

# ============================================================
#  PATCH HELPERS
# ============================================================
def extract_patch(corners):
    if not corners or len(corners) < 4:
        raise ValueError(f"Expected 4 corners, got {len(corners) if corners else 0}")
    return {
        "ul":      corners[0],
        "lr":      corners[2],
        "ll":      corners[3],
        "retries": 0,
    }

# ============================================================
#  SERIAL (TCP) SEND HELPERS
# ============================================================
def send_labeled(label, x, y, expect):
    with socket_lock:
        local_sock = tcp_socket
    if not local_sock:
        return False

    try:
        with socket_lock:
            local_sock.sendall(f"{label} {x} {y}\n".encode())
    except Exception as e:
        print(f"[ERROR] TCP send failed: {e}")
        return False

    deadline = time.time() + SEND_TIMEOUT
    while time.time() < deadline:
        try:
            resp = serial_response_queue.get(timeout=0.05)
            if resp == expect:
                return True
            print(f"[WARN] Unexpected: '{resp}', wanted '{expect}'")
        except queue.Empty:
            continue

    print(f"[ERROR] Timeout waiting for '{expect}' after {label}")
    return False


def send_patch(patch):
    return (
        send_labeled("LL", patch["ll"]["x"], patch["ll"]["y"], "ACK") and
        send_labeled("UL", patch["ul"]["x"], patch["ul"]["y"], "ACK") and
        send_labeled("LR", patch["lr"]["x"], patch["lr"]["y"], "PATCH_READY")
    )


def stop_robot():
    with socket_lock:
        local_sock = tcp_socket
    if local_sock:
        try:
            with socket_lock:
                local_sock.sendall(b"STOP\n")
            print("[SAFETY] STOP sent")
        except Exception:
            pass

# ============================================================
#  TCP LISTENER THREAD  (replaces serial_listener)
# ============================================================
def tcp_listener():
    global tcp_socket, robot_state

    while True:
        with socket_lock:
            local_sock = tcp_socket

        if not local_sock:
            try_reconnect_esp32()
            continue

        try:
            line = _read_line_from_socket(local_sock, timeout=None)
            if not line:
                raise ConnectionResetError("Empty read — socket closed")

            if line == "DONE":
                with state_lock:
                    robot_state = "DONE"
                print("[TCP] DONE received")

            elif line == "ALIVE":
                pass  # heartbeat — silently ignored

            else:
                serial_response_queue.put(line)

        except Exception as e:
            print(f"[TCP] Disconnected: {e}")
            with socket_lock:
                try:
                    tcp_socket.close()
                except Exception:
                    pass
                tcp_socket = None
            with state_lock:
                robot_state = "DISCONNECTED"

# ============================================================
#  SCHEDULER THREAD  (unchanged logic)
# ============================================================
def scheduler():
    global robot_state, current_patch

    while True:
        action     = None
        work_patch = None

        with state_lock:

            if robot_state == "MOVING":
                if not cv_connected.is_set() or not is_position_fresh():
                    action       = "stop"
                    robot_state  = "IDLE"
                    current_patch = None

            elif robot_state == "IDLE":
                if not patch_queue.empty() and cv_connected.is_set():
                    current_patch = patch_queue.get_nowait()
                    robot_state   = "SENDING"
                    action        = "send"
                    work_patch    = current_patch

            elif robot_state == "SENDING":
                if current_patch is not None:
                    action     = "send"
                    work_patch = current_patch

            elif robot_state == "DONE":
                action        = "verify"
                work_patch    = current_patch
                current_patch = None
                robot_state   = "IDLE"

        if action == "stop":
            stop_robot()

        elif action == "send" and work_patch is not None:
            success = send_patch(work_patch)
            with state_lock:
                if success:
                    robot_state = "MOVING"
                    print("[INFO] Robot moving...")
                else:
                    work_patch["retries"] += 1
                    if work_patch["retries"] < MAX_PATCH_RETRIES:
                        print(f"[WARN] Send failed, requeueing (attempt {work_patch['retries']}/{MAX_PATCH_RETRIES})")
                        patch_queue.put(work_patch)
                    else:
                        print(f"[ERROR] Patch failed {MAX_PATCH_RETRIES}x — discarding")
                    current_patch = None
                    robot_state   = "IDLE"

        elif action == "verify" and work_patch is not None:
            pos = get_robot_position()
            if not cv_connected.is_set() or not is_position_fresh():
                print("[WARN] CV unavailable — skipping verify")
            else:
                dist = math.hypot(
                    pos["x"] - work_patch["lr"]["x"],
                    pos["y"] - work_patch["lr"]["y"]
                )
                print("[OK] Target reached" if dist < 2.0 else f"[WARN] Off target by {dist:.2f}cm")

        time.sleep(0.1)


threading.Thread(target=tcp_listener, daemon=True).start()
threading.Thread(target=scheduler,    daemon=True).start()

# ============================================================
#  FASTAPI ENDPOINTS  (unchanged)
# ============================================================
@app.websocket("/ws/robot_position")
async def robot_position_ws(websocket: WebSocket):
    global _robot_position

    await websocket.accept()
    cv_connected.set()
    print("[WS] CV CONNECTED")

    last_log = time.time()
    try:
        while True:
            data = await websocket.receive_json()
            with position_lock:
                _robot_position = {
                    "x":     data["x"],
                    "y":     data["y"],
                    "angle": data["angle"],
                    "t":     time.time(),
                }
            if time.time() - last_log > 2:
                print(f"[WS] pos x={data['x']:.2f} y={data['y']:.2f} ang={data['angle']:.1f}")
                last_log = time.time()

    except Exception as e:
        cv_connected.clear()
        print(f"[WS] CV DISCONNECTED | {e}")


@app.post("/patches")
def receive_patches(data: dict):
    added, dropped = 0, 0
    for p in data.get("patches", []):
        if patch_queue.qsize() >= MAX_QUEUE_SIZE:
            dropped += 1
            continue
        try:
            patch_queue.put(extract_patch(p["corners"]))
            added += 1
        except (ValueError, KeyError) as e:
            print(f"[ERROR] Bad patch: {e}")
            dropped += 1
    return {"queued": added, "dropped": dropped, "queue_size": patch_queue.qsize()}


@app.get("/status")
def status():
    pos = get_robot_position()
    with state_lock:
        return {
            "robot_state":      robot_state,
            "current_patch":    current_patch,
            "patches_remaining": patch_queue.qsize(),
            "cv_connected":     cv_connected.is_set(),
            "robot_position":   pos,
        }


@app.get("/ready")
def ready():
    with state_lock:
        return True
        return {"ready": robot_state == "IDLE" and patch_queue.empty()}