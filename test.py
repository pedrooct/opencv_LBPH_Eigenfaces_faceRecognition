import cv2
import os
import numpy as np
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('-i', "--image", help="Path to the test image", required=True)
parser.add_argument('-c', "--classes", help="Path to the classes", required=True)
parser.add_argument('-w', '--width', help="width to resize",default=100,type=int)
parser.add_argument('-e', '--height', help="height to resize",default=100,type=int)
parser.add_argument('-a', '--algorithm', help="1-LBPH, 2-Eigenfaces",default=1,type=int)
args = vars(parser.parse_args())

test_img = cv2.imread(args["image"],0)
test_img=cv2.resize(test_img,(args["width"],args["height"]))
if args["algorithm"]==1:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
elif args["algorithm"]==2:
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
else:
    print("Error, not valid")
    exit(0)

face_recognizer.read('trainingData.yml')
name = {}
file = open(args["classes"], "r")
for i,line in enumerate(file):
    name[i+1]=line.strip()
file.close()
print(name)
label, confidence = face_recognizer.predict(test_img)
print("label: ", label)
print("confidence: ", confidence)

predicted_name = name[label]

if confidence > 50:
    predicted_name = "unknown, "
    predicted_name += str(confidence)

print(predicted_name)

cv2.waitKey(0)
cv2.destroyAllWindows()
