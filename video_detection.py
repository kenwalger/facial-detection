import argparse
import config
import cv2

from facedetector import FaceDetector
import imutils

# setup the argument parse for command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--face", required=True,
                    help="path to face cascade classifier ")
parser.add_argument("-v", "--video", help="path to video (optional)")

arguments = vars(parser.parse_args())

detector = FaceDetector(arguments["face"])

# detect if a video file is passed in, otherwise use the camera
if not arguments.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(arguments["video"])

# continue processing until the passed in video file is done, or the user
# stops the application by pressing the 'q' key
while True:
    (grabbed, frame) = camera.read()
    # read() returns a boolean value of success and the frame

    # if nothing is returned don't keep going
    if arguments.get("video") and not grabbed:
        break

    # resize the frame to 300 pixels in width
    frame = imutils.resize(frame, width=300)
    # convert the frame to gray scale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # run the frame through the facial detector in the same fashion as
    # still pictures from picture_detection.py
    face_rectangles = detector.detect(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=config.WINDOW_SIZE)

    # Make a copy of the the frame... just in case
    frame_clone = frame.copy()

    # Loop over the bounding boxes and draw rectangles
    for (fX, fY, fW, fH) in face_rectangles:
        cv2.rectangle(frame_clone,
                      (fX, fY),
                      (fX + fW, fY + fH),
                      config.GREEN_BOX,
                      config.LINE_THICKNESS)

    # Show off our work to the world
    cv2.imshow("Face", frame_clone)

    # Provide a way to stop the display
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean everything up
camera.release()
cv2.destroyAllWindows()
