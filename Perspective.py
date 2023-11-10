import numpy as np
import cv2

#cap = cv2.VideoCapture("test1.mp4")
#cap = cv2.VideoCapture("test2.mp4")
#cap = cv2.VideoCapture("test3.mp4")
#cap = cv2.VideoCapture("test4.mp4")
cap = cv2.VideoCapture("test5.mp4")

src = np.float32([(550, 460),
                  (150, 720),
                  (1200, 720),
                  (770, 460)])

dst = np.float32([(100, 0),
                  (100, 720),
                  (1100, 720),
                  (1100, 0)])

def front_to_top(img, M):
    size = (1280, 720)
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

def top_to_front(img, M_inverse):
    size = (1280, 720)
    return cv2.warpPerspective(img, M_inverse, size, flags=cv2.INTER_LINEAR)


while True:
    ret, frame = cap.read()

    if not ret:
        cap.release()
        cap = cv2.VideoCapture("test5.mp4")
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    M = cv2.getPerspectiveTransform(src, dst)
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    output_top = front_to_top(frame, M)
    output_front = top_to_front(frame, M_inverse)

    #cv2.imshow("top", output_top)
    cv2.imshow("front", output_front)
    key = cv2.waitKey(1)
    if key == '27':
        break

cap.release()
cv2.destroyAllWindows()

