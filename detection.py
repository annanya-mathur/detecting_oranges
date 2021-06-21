import cv2
import numpy as np

cap=cv2.VideoCapture(0)
while True:
    r,frame=cap.read()
    # decreasing noise

    blur = cv2.GaussianBlur(frame, (35, 35), 0)


    # decreasing noise
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])

    orange_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask=cv2.bitwise_and(frame,frame,mask=orange_mask)
    contours, h = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(contour)

    # finding out maximum area
    contour = max(contours, key=lambda x: cv2.contourArea(x))
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 4)

    cv2.imshow("blur",blur)
    cv2.imshow("frame", frame)
    cv2.imshow("mask",mask)
    if cv2.waitKey(1)==ord('q'):          # cv2.waitKey(10000)  # 1347.5
        break


cap.release()
cv2.destroyAllWindows()