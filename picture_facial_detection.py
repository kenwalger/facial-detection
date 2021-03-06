import argparse
import cv2

from facedetector import FaceDetector


# setup the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f",
                    "--face",
                    required=True,
                    help="path to where the face cascade resides")
parser.add_argument("-i",
                    "--image",
                    required=True,
                    help="path to where the image file resides")
args = vars(parser.parse_args())

# get the image to use and convert it to Grayscale for processing
image = cv2.imread(args["image"])
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# setup our Classifier processing
detector = FaceDetector(args["face"])
face_rectangles = detector.detect(gray_image,
                                  scaleFactor=1.1,
                                  minNeighbors=5,
                                  minSize=(30, 30))

# Print out the number of faces in the image that were found
print("I found {} face(s)".format(len(face_rectangles)))

# Loop through the faces and compute the rectangles to be drawn
for (x, y, w, h) in face_rectangles:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Show the image and wait to close the window until a key is pressed
cv2.imshow("Faces", image)
cv2.waitKey(0)
