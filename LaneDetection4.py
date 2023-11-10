import numpy as np
import cv2
import glob
from collections import deque

cap = cv2.VideoCapture("test1.mp4")


## Choosing Points for Perspective Transformation
top_left = (420, 460) #(480, 460)
bot_left = (150, 720) #(200, 680)
bot_right = (1200, 720) #(1060, 680)
top_right = (760, 460) #(750, 460)


src = np.float32([(550, 460),
                  (150, 720),
                  (1200, 720),
                  (770, 460)])

# Define source and destination points for perspective transformation
src_points = np.float32([top_left, bot_left, bot_right, top_right])
dst_points = np.float32([(100, 0), (100, 720), (1100, 720), (1100, 0)])
M_inv = cv2.getPerspectiveTransform(dst_points, src)


def region_of_interest(img):
    mask = np.zeros_like(img)
    imshape = img.shape
    if len(imshape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #vertices = np.array([[(0, imshape[0]), (imshape[1] * .20, imshape[0] * .38), (imshape[1] * .80, imshape[0] * .38),
    #                      (imshape[1], imshape[0])]], dtype=np.int32)
    vertices = np.array([[(0, imshape[0]), (imshape[1] * .20, imshape[0] * .38), (imshape[1] * .80, imshape[0] * .38),
                          (imshape[1], imshape[0])]], dtype=np.int32)
    # vertices = np.array([[(0, imshape[0]), (imshape[1] * .30, imshape[0] * .48), (imshape[1] * .62, imshape[0] * .48),
                          #(imshape[1], imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def birdseye_transform(image):
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

    return M, M_inv

def front_to_top(img, M):
    size = (1280, 720)
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

def top_to_front(img, M_inverse):
    size = (1280, 720)
    return cv2.warpPerspective(img, M_inverse, size, flags=cv2.INTER_LINEAR)

def getCalibrationParams(images, nx, ny):
    objPoints = []
    imgPoints = []

    objP = np.zeros((ny * nx, 3), np.float32)
    objP[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny),
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        if ret == True:
            imgPoints.append(corners)
            objPoints.append(objP)

    return imgPoints, objPoints


# Calibrate Image
def calibrateImage(calibPath, calibImg):
    nx = 9
    ny = 6
    # images = glob.glob(calibPath)
    images = glob.glob("{}/*".format(calibPath))
    imgPoints, objPoints = getCalibrationParams(images, nx, ny)

    img = cv2.imread(calibImg)
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, img_size, None, None)

    return mtx, dist


def undistort(img, mtx, dist):
    # Convert to grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.undistort(img, mtx, dist, None, mtx)


def color_thresholding(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Yellow lane detection thresholds
    lower_yellow = np.array([15, 80, 120])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # cv2.imshow("yellow", yellow_mask)


    # White lane detection thresholds
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 80, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    # cv2.imshow("white", white_mask)

    # Combine results
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

    return combined_mask

def hls_threshold(img, channel, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 'h':
        channelImg = hls[:,:,0]
    elif channel == 'l':
        channelImg = hls[:,:,1]
    elif channel == 's':
        channelImg = hls[:,:,2]
    hlsBinary = np.zeros_like(channelImg)
    hlsBinary[(channelImg > thresh[0]) & (channelImg <= thresh[1])] = 1
    return hlsBinary


def color_thresholding2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    adaptive_Thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    # HLS S-channel Threshold
    hlsBinary_s = hls_threshold(img, channel='s', thresh=(100, 255))
    # HLS H-channel Threshold
    hlsBinary_h = hls_threshold(img, channel='h', thresh=(10, 40))
    # HLS L-channel Threshold
    hlsBinary_l = hls_threshold(img, channel='l', thresh=(200, 255))
    # Combine channel thresholds
    combined = np.zeros_like(hlsBinary_s)
    combined[((hlsBinary_h == 1) & (hlsBinary_s == 1)) | (hlsBinary_l == 1) | (adaptive_Thresh == 1)] = 1

    return combined


def threshold_rel(img, lo, hi):
    vmin = np.min(img)
    vmax = np.max(img)

    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    return np.uint8((img >= vlo) & (img <= vhi)) * 255


def threshold_abs(img, lo, hi):
    return np.uint8((img >= lo) & (img <= hi)) * 255


def Final_thresholding(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel)

    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 10)

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    v_channel = hsv[:, :, 2]

    right_lane = threshold_rel(l_channel, 0.8, 1.0)
    right_lane[:, :750] = 0

    #left_lane = threshold_abs(h_channel, 20, 30)
    left_lane = threshold_rel(v_channel, 0.7, 1.0)
    left_lane[:, 550:] = 0

    img2 = left_lane | right_lane

    return img2


def canny_edge_detection(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(img, 75, 150)
    return edges

def static_window_search(left_base, right_base, roi_frame):
    # Sliding Window
    y = 400  # 472
    lx = []
    rx = []

    msk = roi_frame.copy()

    while y > 0:
        # Left Threshold
        img = roi_frame[y - 40:y, left_base - 50:left_base + 50]  # 40 * 100 pixel
        # cv2.imshow("iii", img)
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            # cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                # get the Centre of Mass Coord
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # print("left",cx,cy)
                lx.append(left_base - 50 + cx)
                left_base = left_base - 50 + cx

        # Right Threshold
        img = roi_frame[y - 40:y, right_base - 50:right_base + 50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                # get the Centre of Mass Coord
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # print("right", cx, cy)
                rx.append(right_base - 50 + cx)
                right_base = right_base - 50 + cx

        cv2.rectangle(msk, (left_base - 50, 720 - y), (left_base + 50, 720 - (y - 40)), (255, 255, 255), -1)
        cv2.rectangle(msk, (right_base - 50, 720 - y), (right_base + 50, 720 - (y - 40)), (255, 255, 255), -1)
        # cv2.rectangle(msk, (left_base-50, 720- y), (right_base + 50,720-(y-40)), (255, 0, 0), -1)
        y -= 40

    front_pers = top_to_front(msk, M_inv)

    # Combine the result with the original frame
    result = cv2.addWeighted(frame, 1, front_pers, 0.3, 0)

    return result


def pixels_in_window(roi_frame, center, margin, height):
    """ Return all pixel that in a specific window

    Parameters:
        center (tuple): coordinate of the center of the window
        margin (int): half width of the window
        height (int): height of the window

    Returns:
        pixelx (np.array): x coordinates of pixels that lie inside the window
        pixely (np.array): y coordinates of pixels that lie inside the window
    """
    # Nonzero pixels in the image
    nonzero = roi_frame.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    topleft = (center[0]-margin, center[1]-height//2)
    bottomright = (center[0]+margin, center[1]+height//2)

    condx = (topleft[0] <= nonzerox) & (nonzerox <= bottomright[0])
    condy = (topleft[1] <= nonzeroy) & (nonzeroy <= bottomright[1])
    return nonzerox[condx&condy], nonzeroy[condx&condy]

def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)

def measure_curvature(left_fit, right_fit):
    ym = 30 / 720
    xm = 3.7 / 700

    # left_fit = left_fit.copy()
    # right_fit = right_fit.copy()
    y_eval = 700 * ym

    # Compute R_curve (radius of curvature)
    left_curveR = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curveR = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    xl = np.dot(left_fit, [700 ** 2, 700, 1])
    xr = np.dot(right_fit, [700 ** 2, 700, 1])
    pos = (1280 // 2 - (xl + xr) // 2) * xm
    return left_curveR, right_curveR, pos

# Parameters for window search
nwindows = 9  # Number of sliding windows
minpix = 50    # Minimum number of pixels found to recenter window
margin = 100   # Width of the windows

new_margin_value = 150  # Margin value for adjusting lanes separation
parallel_threshold = 0.1  # Threshold for detecting parallel lanes
prev_leftx_base = 0;

# Global variables to store previous lane information
previous_left_fits = deque(maxlen=5)
previous_right_fits = deque(maxlen=5)

# Width of the windows (consistent across all windows)
window_width = 2 * margin

def dynamic_window_search(roi_frame):
    global prev_leftx_base
    # Height of windows
    window_height = np.int32(roi_frame.shape[0] // nwindows) #- 10

    # Nonzero pixels in the image
    nonzero = roi_frame.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    histogram = hist(roi_frame)
    midpoint = histogram.shape[0] // 2

    left_half_histogram = histogram[:midpoint]
    right_half_histogram = histogram[midpoint:]

    # Set the search ranges for the left and right lanes
    left_search_range = (midpoint - 100, midpoint + 100) #200
    right_search_range = (midpoint + 50, midpoint + 200) # 600

    # Find the peaks within the search ranges for the left lane
    left_search_histogram = left_half_histogram[left_search_range[0]:left_search_range[1]]
    if np.any(left_search_histogram):
        left_peak_index = np.argmax(left_search_histogram) + left_search_range[0]
        leftx_base = left_peak_index
        #print("here")
    else:
        # Handle the case where no valid peak is found for the left lane
        leftx_base = np.argmax(histogram[:midpoint])

    # Find the peaks within the search ranges for the right lane
    right_search_histogram = right_half_histogram[right_search_range[0]:right_search_range[1]]
    if np.any(right_search_histogram):
        right_peak_index = np.argmax(right_search_histogram) + right_search_range[0]
        rightx_base = right_peak_index
    else:
        # Handle the case where no valid peak is found for the right lane
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint


    #leftx_base = np.argmax(histogram[:midpoint])

    if (leftx_base != 0):
        prev_leftx_base = leftx_base
    elif(leftx_base == 0):
        leftx_base = prev_leftx_base

    #rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # print(leftx_base, rightx_base)


    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    y_current = roi_frame.shape[0] + window_height // 2

    # Lists to store the lane pixel indices
    # left_lane_inds = []
    # right_lane_inds = []

    # Create empty lists to reveice left and right lane pixel
    leftx, lefty, rightx, righty = [], [], [], []

    # Iterate through windows
    for window in range(nwindows):
        '''
        # Identify window boundaries
        win_y_low = roi_frame.shape[0] - (window + 1) * window_height
        win_y_high = roi_frame.shape[0] - window * window_height

        # Left_Window
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        # Right_Window
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify nonzero pixels within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        print(good_left_inds, good_right_inds)
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If pixels found in current window > minpix, recenter next window
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
            
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        print("left_lane_inds", left_lane_inds)
    
        return left_lane_inds, right_lane_inds
        
        '''
        y_current -= window_height
        center_left = (leftx_current, y_current)
        center_right = (rightx_current, y_current)

        good_left_x, good_left_y = pixels_in_window(roi_frame, center_left, margin, window_height)
        good_right_x, good_right_y = pixels_in_window(roi_frame, center_right, margin, window_height)

        # Append these indices to the lists
        leftx.extend(good_left_x)
        lefty.extend(good_left_y)
        rightx.extend(good_right_x)
        righty.extend(good_right_y)

        '''
        if len(good_left_x) > minpix and len(good_right_x) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_x]))
            rightx_current = np.int32(np.mean(nonzerox[good_right_x]))
        '''

    return leftx, lefty, rightx, righty

def polynomial_fit(leftx, lefty, rightx, righty):
    '''
    # Extract pixel positions
    leftx = roi_frame.nonzero()[1][left_lane_inds]
    lefty = roi_frame.nonzero()[0][left_lane_inds]
    rightx = roi_frame.nonzero()[1][right_lane_inds]
    righty = roi_frame.nonzero()[0][right_lane_inds]
    '''

    # Fit a second-degree polynomial to each lane
    if len(lefty) > 1500:
        left_fit = np.polyfit(lefty, leftx, 2)

    if len(righty) > 1500:
        right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit , lefty, righty

previous_valid_left_fit = np.array([0, 0, 0])
previous_valid_right_fit = np.array([0, 0, 0])

def visualize_lane(frame, roi_frame, leftx, lefty, rightx, righty):

    # Initialize default values for left_fit and right_fit
    left_fit = np.array([0, 0, 0])
    right_fit = np.array([0, 0, 0])

    global previous_valid_left_fit, previous_valid_right_fit, margin

    # If lanes are detected in the current frame, update previous lane information
    if len(leftx) > minpix and len(rightx) > minpix:
        previous_left_fits.append(np.polyfit(lefty, leftx, 2))
        previous_right_fits.append(np.polyfit(righty, rightx, 2))

    # Fit a second-degree polynomial to each lane
    if len(lefty) > 1500 and len(leftx) > minpix:
        previous_left_fits.append(np.polyfit(lefty, leftx, 2))
        #left_fit = np.polyfit(lefty, leftx, 2)
        #previous_valid_left_fit = left_fit

    if len(righty) > 1500 and len(rightx) > minpix:
        previous_right_fits.append(np.polyfit(righty, rightx, 2))
        #right_fit = np.polyfit(righty, rightx, 2)
        #previous_valid_right_fit = right_fit

    # Smooth lane estimates using moving average
    if len(previous_left_fits) > 0 and len(previous_right_fits) > 0:
        left_fit = np.mean(previous_left_fits, axis=0)
        previous_valid_left_fit = left_fit
        right_fit = np.mean(previous_right_fits, axis=0)
        previous_valid_right_fit = right_fit

    if len(lefty) <= 1500:
        left_fit = previous_valid_left_fit

    if len(righty) <= 1500:
        right_fit = previous_valid_right_fit

    # Check if the detected lanes are too parallel
    if np.abs(left_fit[0] - right_fit[0]) < parallel_threshold:
        # Reconfigure the margin to separate lanes
        margin = new_margin_value

    # Create an output image to draw on and visualize the lane
    out_img = np.dstack((roi_frame, roi_frame, roi_frame))  * 255
    # Generate x and y values for plotting
    maxy = roi_frame.shape[0] - 1
    miny = roi_frame.shape[0] // 3

    if len(lefty):
        maxy = max(maxy, np.max(lefty))
        miny = min(miny, np.min(lefty))

    if len(righty):
        maxy = max(maxy, np.max(righty))
        miny = min(miny, np.min(righty))

    # Generate x values for plotting the polynomial
    ploty = np.linspace(miny, maxy, roi_frame.shape[0])
    #ploty = np.linspace(0, roi_frame.shape[0]-1, roi_frame.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Visualization
    for i, y in enumerate(ploty):
        l = int(left_fitx[i])
        r = int(right_fitx[i])
        y = int(y)
        cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

    out_img = top_to_front(out_img, M_inv)
    out_img = cv2.addWeighted(out_img, 1, frame, 0.6, 0)

    '''
    # Create an image to draw the lane lines
    window_img = np.zeros_like(out_img)

    # Color in the lane pixels
    out_img[roi_frame.nonzero()[0], roi_frame.nonzero()[1]] = [0, 255, 0]  # Green

    # Generate a polygon to represent the lane area
    left_lane = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_lane = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    lane_pts = np.hstack((left_lane, right_lane))

    # Draw the lane area onto the warped blank image
    cv2.fillPoly(window_img, np.int_([lane_pts]), (0, 255, 0))

    window_img = top_to_front(window_img, M_inv)

    # Combine the result with the original frame
    result = cv2.addWeighted(frame, 1, window_img, 0.3, 0)
    '''

    return out_img, left_fit, right_fit

def drowsy_check(out_img, left_fit, right_fit):
    np.set_printoptions(precision=6, suppress=True)
    lR, rR, pos =measure_curvature(left_fit, right_fit)
    dir = []

    value = None
    if abs(left_fit[0]) > abs(right_fit[0]):
        value = left_fit[0]
    else:
        value = right_fit[0]

    if abs(value) <= 0.00015:
        dir.append('F')
    elif value < 0:
        dir.append('L')
    else:
        dir.append('R')

    if len(dir) > 10:
        dir.pop(0)

    direction = max(set(dir), key=dir.count)
    msg = "Keep Straight Ahead"
    curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))
    if direction == 'L':
        msg = "Left Curve Ahead"
    if direction == 'R':
        msg = "Right Curve Ahead"

    # if direction == 'F':
    cv2.putText(out_img, msg, org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                thickness=2)
    if direction in 'LR':
        cv2.putText(out_img, curvature_msg, org=(10, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(255, 255, 255), thickness=2)

    return out_img

mtx, dist = calibrateImage('camera_cal', 'chess.png')

while True:
    ret, frame = cap.read()

    if not ret:
        cap = cv2.VideoCapture("test5.mp4")
        continue

    frame = undistort(frame, mtx, dist)

    '''
    cv2.circle(frame, top_left, 5, (0, 0, 255), -1)
    cv2.circle(frame, top_right, 5, (0, 0, 255), -1)
    cv2.circle(frame, bot_left, 5, (0, 0, 255), -1)
    cv2.circle(frame, bot_right, 5, (0, 0, 255), -1)
    '''
    #cv2.imshow("Frame ", frame)
    M = birdseye_transform(frame)

    top_bev = front_to_top(frame, M[0])
    cv2.imshow("top_bev", top_bev)
    thresh =  Final_thresholding(top_bev)
    #cv2.imshow("thresh", thresh)
    hls_thresh = color_thresholding(top_bev)
    #cv2.imshow("hls_thresh", hls_thresh)
    combined_thresh = thresh | hls_thresh
    #cv2.imshow("combined_thresh", combined_thresh)

    roi_frame = region_of_interest(combined_thresh)
    #frame_c = region_of_interest(frame)
    cv2.imshow("ROI", roi_frame)
    #cv2.imshow("ROI Check", frame_c)
    canny_mask = canny_edge_detection(roi_frame)
    #cv2.imshow("canny_mask", canny_mask)

    # Dynamic Window Search
    # left_lane_inds, right_lane_inds  = dynamic_window_search(roi_frame, left_base, right_base)
    leftx, lefty, rightx, righty = dynamic_window_search(roi_frame)

    # Polynomial Fitting
    # left_fit1, right_fit1, lefty1, righty1 = polynomial_fit(leftx, lefty, rightx, righty)

    # Visualization
    vis_out, left_fit, right_fit = visualize_lane(frame, roi_frame, leftx, lefty, rightx, righty)
    result = drowsy_check(vis_out, left_fit, right_fit)

    cv2.imshow("result ", result)

    '''
    cv2.imshow("combined ", msk)
    final = frame | front_pers
    #cv2.imshow("frame", frame)
    cv2.imshow("Sliding Window", final)    
    '''

    '''
    cv2.imshow("frame", birdseye)
    cv2.imshow("canny", canny_mask)
    cv2.imshow("thresh", thresholded_mask)
    '''

    key = cv2.waitKey(1)
    if key == '27':
        break

cap.release()
cv2.destroyAllWindows()

