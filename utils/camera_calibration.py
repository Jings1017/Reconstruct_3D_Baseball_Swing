if __name__ == "__main__":
    import cv2
    import argparse
    from os import makedirs
    from os.path import exists
    from pathlib import Path
    import numpy as np
    import cv2
    import glob
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input video")
    parser.add_argument("--output", type=str, help="output frame folder")
    parser.add_argument(
        "--chessboard_size", nargs="+", type=int, help="chessboard_size"
    )
    args = parser.parse_args()

    chessboard_size = tuple(args.chessboard_size)
    input_folder = args.input

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(
        -1, 2
    )
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(input_folder + "/*.png")
    success_frame = []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("Reading: ", fname)
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            base = os.path.basename(fname)
            success_frame.append(os.path.splitext(base)[0])
            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow("img", img)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )

            np.savez(
                input_folder + "/camera_calibration2",
                ret=ret,
                mtx=mtx,
                dist=dist,
                rvecs=rvecs,
                tvecs=tvecs,
            )
    print(mtx)
    print(dist)
    cv2.destroyAllWindows()
    print(sorted(success_frame))
