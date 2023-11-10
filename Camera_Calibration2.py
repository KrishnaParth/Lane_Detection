import numpy as np
import cv2
# To read all images in one go
import glob

#
objpoints = []
imgpoints = []

path = "Advanced-Lane-Lines/camera_cal/"

def undistort(distorted_image):
    return cv2.undistort(distorted_image, mtx, dist, None, mtx)

images = glob.glob("{}/*".format("camera_cal"))
# print(images)

# chess : 10 col, 7 rows, Creating 3D Array
objp = np.zeros((9*6, 3), np.float32)
#print(objp)

# Create mesh grid on objp array in 3-D space
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
#print(objp[:, :2])

for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img, (9, 6))

    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)

shape = (img.shape[1], img.shape[0])
ret, mtx, dist, _,_ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

img = cv2.imread("camera_cal/calibration1.jpg")
output = undistort(img)

#cv2.imshow("orig",img)
#cv2.imshow("res",output)

display = np.hstack((img, output))
cv2.imshow("comp",display)





cv2.waitKey(0)
cv2.destroyAllWindows()