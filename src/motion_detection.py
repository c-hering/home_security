import datetime
import imutils
from imutils.object_detection import non_max_suppression
import argparse
from time import sleep
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

capture = cv2.VideoCapture(0)
sleep(.25)

firstFrame = None

while True:
    (grabbed, frame) = capture.read()
    text = "No Motion"

    if not grabbed:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray
        continue

    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            continue

        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cnts = np.array([[x, y, x + w, y + h] for (x, y, w, h) in cnts])
    pick = non_max_suppression(cnts, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
	    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.putText(frame, "Room Status: {}".format(text), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)

    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()
