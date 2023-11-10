import numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt


def canny_edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 150)
    return edges


def region_of_interest(img):
    height = img.shape[0]
    triangle = np.array([[(200, height), (1000, height), (550, 250)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, np.int32([triangle]), 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def display_lines(img, lines):
    line_mask = np.zeros_like(img)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_mask, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return line_mask


def make_coordinates(image, line_parameters):
    if hasattr(line_parameters, "__iter__") and len(line_parameters) == 2:
        slope, intercepts = line_parameters
    else:
        slope, intercepts = 0, 0  # Assign default values if line_parameters is not iterable

    # slope, intercepts = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))  # lines will go 3/5th way upward

    if slope is not 0:
        x1 = int((y1 - intercepts)/slope)
        x2 = int((y2 - intercepts) / slope)
    else:
        x1 = 0
        x2 = 0
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            # print("parame", parameters)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        # print("left_fit", left_fit)
        # print("right_fit", right_fit)
        left_fit_avg = np.average(left_fit, axis=0)
        right_fit_avg = np.average(right_fit, axis=0)

        left_line = make_coordinates(img, left_fit_avg)
        right_line = make_coordinates(img, right_fit_avg)

        slope_left, _ = left_fit_avg
        slope_left = round(slope_left, 3)
        slope_right, _ = right_fit_avg
        slope_right = round(slope_right, 3)

        print(slope_left, slope_right)

        if slope_left < -2:
            cv2.putText(frame, 'TURN RIGHT',
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2, cv2.LINE_4)

        elif slope_right > 1.5:
            cv2.putText(frame, 'TURN LEFT',
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2, cv2.LINE_4)

        return numpy.array([left_line, right_line])
    else:
        return numpy.array([])

cap = cv2.VideoCapture("test5.mp4")
#cap = cv2.VideoCapture("test1.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        cap.release()
        cap = cv2.VideoCapture("test5.mp4")
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    canny_img = canny_edge_detection(frame)

    # Find out Polygon using Matplotlib
    # plt.imshow(frame)
    # plt.show()

    cropped_img = region_of_interest(canny_img)

    lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 10, np.array([]),
                            minLineLength=30, maxLineGap=5)

    # Optimisation
    averaged_lines = average_slope_intercept(frame, lines)

    for line in averaged_lines:
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / (x2 - x1)
        print(slope)

    '''
        if slope < -deviation_threshold:
            print("Going away from left lane.")
            # Take appropriate action, like alerting the driver

        elif slope > deviation_threshold:
            print("Going away from right lane.")
            # Take appropriate action, like alerting the driver
    '''
    line_mask = display_lines(frame, averaged_lines)
    #print(line_mask)

    # To increase Image Definition
    Final_img = cv2.addWeighted(frame, 0.8, line_mask, 1, 1)

    cv2.imshow("frame", Final_img)
    #cv2.imshow("mask", line_mask)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

