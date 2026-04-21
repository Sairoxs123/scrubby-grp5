import cv2
import numpy as np
import math

# ==========================================
# CONSTANTS
# ==========================================
BOARD_WIDTH_CM = 105.0
BOARD_HEIGHT_CM = 75.0
RX = BOARD_WIDTH_CM / 1000.0
RY = BOARD_HEIGHT_CM / 700.0

# Mechanical Offsets
OFFSET_FORWARD_CM = -6.5
OFFSET_RIGHT_CM = 1.5

def extract_whiteboard(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None and len(ids) >= 4:
        m = {ids.flatten()[i]: np.mean(corners[i][0], axis=0) for i in range(len(ids))}
        if all(i in m for i in [0,1,2,3]):
            s = np.array([m[0], m[3], m[1], m[2]], dtype="float32")
            d = np.array([[0,0], [1000,0], [0,700], [1000,700]], dtype="float32")
            return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(s, d), (1000, 700))
    return None

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dict_aruco, cv2.aruco.DetectorParameters())

    print("--- ROBOT TRACKING VERIFIER ---")
    print("Origin (0,0) is BOTTOM-LEFT.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        warped = extract_whiteboard(frame)
        if warped is not None:
            # Draw visual Origin at Bottom Left
            cv2.circle(warped, (20, 680), 10, (0, 255, 0), -1)
            cv2.putText(warped, "(0, 0) Origin", (35, 685), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            corners, ids, _ = detector.detectMarkers(warped)

            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    if mid == 42:
                        cv2.aruco.drawDetectedMarkers(warped, corners, ids)

                        c = corners[i][0]
                        mx_px, my_px = np.mean(c[:, 0]), np.mean(c[:, 1])

                        dx = (c[0][0]+c[1][0])/2 - (c[3][0]+c[2][0])/2
                        dy = (c[3][1]+c[2][1])/2 - (c[0][1]+c[1][1])/2
                        theta = math.atan2(dy, dx)

                        # Apply Offsets
                        tx = (mx_px * RX) + (OFFSET_FORWARD_CM * math.cos(theta) - OFFSET_RIGHT_CM * math.sin(theta))
                        ty = (my_px * RY) + (OFFSET_FORWARD_CM * math.sin(theta) + OFFSET_RIGHT_CM * math.cos(theta))

                        # INVERT Y FOR BOTTOM LEFT
                        ty = BOARD_HEIGHT_CM - ty

                        angle_degs = np.degrees(theta) % 360

                        # Draw True Center dot
                        rx_px = int(tx / RX)
                        ry_px = int((BOARD_HEIGHT_CM - ty) / RY)
                        cv2.circle(warped, (rx_px, ry_px), 6, (255, 0, 255), -1)

                        # Put text on screen
                        info = f"X: {tx:.1f} cm | Y: {ty:.1f} cm | Ang: {angle_degs:.0f} deg"
                        cv2.putText(warped, info, (rx_px + 15, ry_px), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        print(info)

            cv2.imshow("Tracking View", warped)

        cv2.imshow("Raw Cam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()