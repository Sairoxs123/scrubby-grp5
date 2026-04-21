from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager
import serial
import threading
import queue
import math
import time

ser = None
robot_state = "IDLE"
robot_position = {"x": 0, "y": 0, "angle": 0, "t": 0}
patch_queue = []
current_patch = None
cv_connected = threading.Event()

SEND_TIMEOUT = 5
serial_lock = threading.Lock()
serial_response_queue = queue.Queue()
state_lock = threading.Lock()
def find_esp32():
    try:
        s = serial.Serial("COM13", 115200, timeout=1, write_timeout=1)
        time.sleep(2)
        s.write(b"HELLO_ROBOT\n")
        response = s.readline().decode().strip()
        print(f"[DEBUG] Response: '{response}'")
        if response == "ROBOT_READY":
            return s
        s.close()
    except Exception as e:
        print(f"[ERROR] {e}")
    return None

@asynccontextmanager
async def lifespan(app):
    global ser, robot_state
    ser = find_esp32()
    if ser:
        robot_state = "IDLE"
        print(f"[OK] Connected to ESP32 on {ser.port}")
    else:
        robot_state = "DISCONNECTED"
        print("[FAIL] ESP32 not found")
    yield

app = FastAPI(lifespan=lifespan)

def extract_patch(corners):
    return {
        "ul": corners[0],
        "lr": corners[2],
        "ll": corners[3],
    }

def send_labeled(label, x, y, expect):
    with serial_lock:
        msg = f"{label} {x} {y}\n"
        try:
            ser.write(msg.encode())
        except Exception as e:
            print(f"[ERROR] Serial write failed: {e}")
            return False

    deadline = time.time() + SEND_TIMEOUT
    while time.time() < deadline:
        try:
            response = serial_response_queue.get(timeout=0.05)
            if response == expect:
                return True
            else:
                print(f"[WARN] Unexpected response: '{response}', expected '{expect}'")
        except queue.Empty:
            continue
    print(f"[ERROR] Timeout waiting for '{expect}' after sending {label}")
    return False

def send_patch(patch):
    if not send_labeled("LL", patch["ll"]["x"], patch["ll"]["y"], "ACK"):
        return False
    if not send_labeled("UL", patch["ul"]["x"], patch["ul"]["y"], "ACK"):
        return False
    if not send_labeled("LR", patch["lr"]["x"], patch["lr"]["y"], "PATCH_READY"):
        return False
    return True

def serial_listener():
    global robot_state, ser

    while True:
        try:
            if ser and ser.in_waiting:
                msg = ser.readline().decode().strip()
                if not msg:
                    pass
                elif msg == "DONE":
                    with state_lock:
                        robot_state = "DONE"
                    print("[SERIAL] Received DONE")
                else:
                    serial_response_queue.put(msg)
        except serial.SerialException:
            ser = None
            with state_lock:
                robot_state = "DISCONNECTED"
            print("[SERIAL] Disconnected")
        time.sleep(0.02)

def is_position_fresh(timeout=0.5):
    return time.time() - robot_position.get("t", 0) < timeout

def stop_robot():
    try:
        if ser:
            with serial_lock:
                ser.write(b"STOP\n")
                print("[SAFETY] STOP sent")
    except:
        pass

def scheduler():
    global robot_state, patch_queue, current_patch

    while True:
        with state_lock:
            state = robot_state

        if state == "MOVING" and (not cv_connected.is_set() or not is_position_fresh()):
            print("[SAFETY] Lost CV → stopping robot")
            stop_robot()
            with state_lock:
                robot_state = "IDLE"
                current_patch = None

        elif state == "IDLE":
            with state_lock:
                if patch_queue and cv_connected.is_set():
                    current_patch = patch_queue.pop(0)
                    robot_state = "SENDING"
                elif patch_queue and not cv_connected.is_set():
                    print("[WAIT] Waiting for CV...")

            with state_lock:
                state = robot_state  # re-read after potential update

            if state == "SENDING":
                print(f"[INFO] Sending patch: {current_patch}")
                if send_patch(current_patch):
                    with state_lock:
                        robot_state = "MOVING"
                    print("[INFO] Robot moving...")
                else:
                    print("[ERROR] Failed to send patch, requeueing")
                    with state_lock:
                        patch_queue.insert(0, current_patch)
                        current_patch = None
                        robot_state = "IDLE"

        elif state == "DONE":
            with state_lock:
                patch = current_patch

            if not cv_connected.is_set() or not is_position_fresh():
                print("[WARN] CV unavailable — cannot verify position")
            else:
                dist = math.hypot(
                    robot_position["x"] - patch["lr"]["x"],
                    robot_position["y"] - patch["lr"]["y"]
                )
                if dist < 2.0:
                    print("[OK] Target reached (verified by CV)")
                else:
                    print(f"[WARN] DONE but off target (error={dist:.2f} cm)")

            with state_lock:
                current_patch = None
                robot_state = "IDLE"

        time.sleep(0.1)

threading.Thread(target=serial_listener, daemon=True).start()
threading.Thread(target=scheduler, daemon=True).start()


@app.websocket("/ws/robot_position")
async def robot_position_ws(websocket: WebSocket):
    global robot_position

    await websocket.accept()
    cv_connected.set()
    print("[WS] CV CONNECTED")

    first_packet = True
    last_log_time = time.time()

    try:
        while True:
            data = await websocket.receive_json()

            # Atomic dict reassignment — prevents half-updated reads
            robot_position = {
                "x": data["x"],
                "y": data["y"],
                "angle": data["angle"],
                "t": time.time()
            }

            if first_packet:
                print(f"[WS] First data: {data}")
                first_packet = False

            if time.time() - last_log_time > 2:
                print(f"[WS] Position: {robot_position}")
                last_log_time = time.time()

    except Exception as e:
        cv_connected.clear()
        print(f"[WS] CV DISCONNECTED | {e}")

@app.post("/patches")
def receive_patches(data: dict):
    global patch_queue
    for p in data["patches"]:
        patch_queue.append(extract_patch(p["corners"]))
    return {"queued": len(patch_queue)}

@app.get("/status")
def status():
    return {
        "robot_state": robot_state,
        "robot_position": robot_position,
        "current_patch": current_patch,
        "patches_remaining": len(patch_queue),
        "cv_connected": cv_connected.is_set()
    }

@app.get("/ready")
def ready():
    return True
    return {
        "ready": robot_state == "IDLE" and len(patch_queue) == 0
    }