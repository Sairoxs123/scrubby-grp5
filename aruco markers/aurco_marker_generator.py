import cv2
# Load the 4x4 dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# Generate marker with ID 0
marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 200)
cv2.imwrite("marker0.png", marker_img)
