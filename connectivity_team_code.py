from fastapi import FastAPI
import serial
import threading
import math
import time

app = FastAPI()

ser = serial.Serial("COM18", 115200, timeout=1)
robot_state = "IDLE"
robot_position = {"x": 0, "y": 0}

patch_queue = []
current_patch = None

SEND_TIMEOUT = 5


@app.on_event("startup")
def connect_robot():
    global ser
    global robot_state
    ser = find_esp32()
    if ser:
        robot_state = "IDLE"
    else:
        robot_state = "DISCONNECTED"


def extract_patch(corners):

    min_x = min(c["x"] for c in corners)
    max_x = max(c["x"] for c in corners)
    min_y = min(c["y"] for c in corners)
    max_y = max(c["y"] for c in corners)

    return {
        "ll": {"x": min_x, "y": min_y},
        "ul": {"x": min_x, "y": max_y},
        "lr": {"x": max_x, "y": min_y}
    }


def send_labeled(label, x, y, expect):

    global ser

    msg = f"{label} {x} {y}\n"
    ser.write(msg.encode())

    deadline = time.time() + SEND_TIMEOUT

    while time.time() < deadline:

        if ser.in_waiting:
            r = ser.readline().decode().strip()
            if r == expect:
                return True

        time.sleep(0.05)

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
    global robot_state
    global ser
    while True:
        try:
            if ser and ser.in_waiting:
                msg = ser.readline().decode().strip()
                if msg == "DONE":
                    robot_state = "DONE"
        except serial.SerialException:
            ser = None
            robot_state = "DISCONNECTED"
        time.sleep(0.05)


def scheduler():

    global robot_state
    global patch_queue
    global current_patch

    while True:

        if robot_state == "IDLE" and patch_queue:

            current_patch = patch_queue.pop(0)

            if send_patch(current_patch):
                robot_state = "MOVING"
            else:
                patch_queue.insert(0,current_patch)
                current_patch = None
                robot_state = "IDLE"


        if robot_state == "DONE" and current_patch:

            robot_position["x"] = current_patch["lr"]["x"]
            robot_position["y"] = current_patch["lr"]["y"]

            current_patch = None
            robot_state = "IDLE"


        time.sleep(0.1)


threading.Thread(target=serial_listener,daemon=True).start()
threading.Thread(target=scheduler,daemon=True).start()


@app.post("/patches")
def receive_patches(data: dict):

    global patch_queue

    for p in data["patches"]:
        patch_queue.append(
            extract_patch(p["corners"])
        )

    return {
        "queued": len(patch_queue)
    }


@app.get("/status")
def status():
    return {
        "robot_state": robot_state,
        "robot_position": robot_position,
        "current_patch": current_patch,
        "patches_remaining": len(patch_queue)
    }


@app.get("/ready")
def ready():
    return {
        "ready": robot_state == "IDLE" and len(patch_queue) == 0
    }