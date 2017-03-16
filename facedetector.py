import cv2


class FaceDetector:
    def __init__(self, faceCascadePath):

        # Define the classifier
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

    def detect(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        """
        :param image: The image to be processed
        :param scaleFactor: How much the image is reduced at each image scale.
        :param minNeighbors: How many neighbors each window will be detected
        :param minSize: width and height of the window size
        :return: List of tuples of the bounding boxes of the face image
        """
        rectangle = self.faceCascade.detectMultiScale(image,
                                                      scaleFactor=scaleFactor,
                                                      minNeighbors=minNeighbors,
                                                      minSize=minSize,
                                                      flags=cv2.CASCADE_SCALE_IMAGE)

        return rectangle
