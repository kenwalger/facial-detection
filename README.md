Facial Detection with OpenCV for Python
====

Presented at Boise Code Camp
18 March 2017

This project uses, and includes, the [haarcascade_frontalface_default](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) 
classifier
#
Useage

Static Images  
`python picture_facial_detection.py --face 
cascades/haarcascade_frontalface_default
.xml 
--image images/single.jpg`
#  
  
Video Images:

With computer camera:  

`python video_facial_detection.py --face 
cascades/haarcascade_frontalface_default.xml`

To process an existing video file:  

`python video_facial_detection.py --face 
cascades/haarcascade_frontalface_default.xml --video <path to video file>`






