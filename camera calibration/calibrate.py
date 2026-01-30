import numpy as np
import cv2
import glob
import argparse
import os

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate(dirpath, prefix, image_format, square_size, width=9, height=6):
    """ Apply camera calibration operation for images in the given directory path. """

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    # This prepares the theoretical "perfect" grid coordinates
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    # Load images
    images = glob.glob(dirpath+'/' + prefix + '*.' + image_format)

    # Check if images were actually found
    if not images:
        print(f"Error: No images found in {dirpath} with prefix '{prefix}' and format '{image_format}'")
        return None

    found_count = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Optional: Draw and display the corners to verify detection (commented out for speed)
            # img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(100)
            found_count += 1

    print(f"Calibration based on {found_count} valid images out of {len(images)} total.")

    if found_count == 0:
        print("\nError: Calibration failed because no valid chessboard patterns were found.")
        print("Possible reasons:")
        print(f"1. The grid size (width={width}, height={height}) might be incorrect.")
        print("   - Ensure you are counting INNER corners, not squares.")
        print("   - Inner corners = (squares_width - 1) x (squares_height - 1).")
        print("2. The images might be too blurry, dark, or the board is not fully visible.")
        print("3. Try swapping width and height.")
        return None

    # Perform the actual calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # ---------------------------------------------------------
    # TEST 1: MATHEMATICAL CHECK (Reprojection Error)
    # ---------------------------------------------------------
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"\n>>> Total Reprojection Error: {mean_error/len(objpoints):.4f} pixels")

    return [ret, mtx, dist, rvecs, tvecs]

def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.release()

def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    cv_file.release()
    return [camera_matrix, dist_matrix]

def visualize_distortion(mtx, dist, image_path):
    """
    TEST 2: VISUAL CHECK
    Loads an image, undistorts it, and displays them side-by-side.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image for visualization.")
        return

    h,  w = img.shape[:2]

    # Refine the camera matrix (this handles the "zoom" or "crop" effect of undistortion)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # Undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image (optional, based on ROI)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # Resize for display if the images are huge
    display_w = 640
    scale = display_w / img.shape[1]
    dim = (display_w, int(img.shape[0] * scale))

    img_small = cv2.resize(img, dim)
    dst_small = cv2.resize(dst, dim)

    # Combine side-by-side
    combined = np.hstack((img_small, dst_small))

    print("\n>>> Showing Visual Test...")
    print("    Left: Original (Curved) | Right: Undistorted (Straight)")
    print("    Press any key in the window to close it.")

    cv2.imshow('Calibration Check: Original vs Undistorted', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--image_dir', type=str, required=True, help='image directory path')
    parser.add_argument('--image_format', type=str, required=True, help='image format, png/jpg')
    parser.add_argument('--prefix', type=str, required=True, help='image prefix')
    parser.add_argument('--square_size', type=float, required=True, help='chessboard square size')
    parser.add_argument('--width', type=int, required=True, help='chessboard width size, default is 9')
    parser.add_argument('--height', type=int, required=True, help='chessboard height size, default is 6')
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')

    args = parser.parse_args()

    # 1. RUN CALIBRATION
    calibration_result = calibrate(
        args.image_dir,
        args.prefix,
        args.image_format,
        args.square_size,
        args.width,
        args.height
    )

    if calibration_result is None:
        exit(1)

    ret, mtx, dist, rvecs, tvecs = calibration_result

    # 2. SAVE RESULTS
    save_coefficients(mtx, dist, args.save_file)
    print(f"Calibration Matrix saved to {args.save_file}")

    # 3. RUN VISUAL CHECK (using the first image in the folder)
    test_images = glob.glob(args.image_dir + '/' + args.prefix + '*.' + args.image_format)
    if test_images:
        visualize_distortion(mtx, dist, test_images[0])