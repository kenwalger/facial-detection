from __future__ import print_function
from facedetector import FaceDetector

import argparse
import cv2

# setup the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="path to where the face "
                                                    "cascade resides")
ap.add_argument("-i", "--image", required=True, help="path to where the image "
                                                     "file resides")
args = vars(ap.parse_args())

# get the image to use and convert it to Grayscale for processing
image = cv2.imread(args["image"])
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# setup our Classifier processing
fd = FaceDetector(args["face"])
face_rectangles = fd.detect(gray_image, scaleFactor=1.1, minNeighbors=5,
                            minSize=(30, 30))

# Print out the number of faces in the image that were found
print("I found {} face(s)".format(len(face_rectangles)))

# Loop through the faces and compute the rectangles to be drawn
for (x, y, w, h) in face_rectangles:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # GRB format

# Show the image and wait to close the window until a key is pressed
cv2.imshow("Faces", image)
cv2.waitKey(0)
