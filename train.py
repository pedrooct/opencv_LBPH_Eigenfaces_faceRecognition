import cv2
import os
import numpy as np
import faceRecognition as fr


import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('-w', '--width', help="width to resize",default=100,type=int)
parser.add_argument('-e', '--height', help="height to resize",default=100,type=int)
parser.add_argument('-a', '--algorithm', help="1-LBPH, 2-Eigenfaces",default=1,type=int)
args = vars(parser.parse_args())

faces, faceID = fr.labels_for_training_data('Dataset/train',args["width"],args["height"])

if args["algorithm"]==1:
    face_recognizer = fr.train_classifierLBPH(faces, faceID)
elif args["algorithm"]==2:
    face_recognizer = fr.train_classifierEigen(faces, faceID)
else:
    print("Option not valid")
    exit(0)

face_recognizer.save("trainingData.yml")
