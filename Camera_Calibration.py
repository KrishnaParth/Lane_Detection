import numpy as np
import cv2

img = cv2.imread("chess.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

nx = 7
ny = 7
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

if ret == True:
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

cv2.imshow("Img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()