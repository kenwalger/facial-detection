import argparse
import config
import cv2

from facedetector import FaceDetector
import imutils

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--face", required=True,
                    help="path to face cascade classifier ")
parser.add_argument("-v", "--video", help="path to video (optional)")

arguments = vars(parser.parse_args())

detector = FaceDetector(arguments["face"])

if not arguments.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(arguments["video"])

while True:
    (grabbed, frame) = camera.read()

    if arguments.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_rectangles = detector.detect(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=config.WINDOW_SIZE)
    frame_clone = frame.copy()

    for (fX, fY, fW, fH) in face_rectangles:
        cv2.rectangle(frame_clone,
                      (fX, fY),
                      (fX + fW, fY + fH),
                      config.BOX_COLOR,
                      config.LINE_THICKNESS)

    cv2.imshow("Face", frame_clone)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
