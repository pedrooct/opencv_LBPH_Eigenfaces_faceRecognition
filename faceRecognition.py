import cv2
import os
import numpy as np


def labels_for_training_data(directory,width,heigth):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping System Files")
                continue
            id=os.path.basename(path)
            img_path=os.path.join(path, filename)
            test_img=cv2.imread(img_path,0)
            if test_img is None:
                print("Image not loaded properly")
            else:
                print("img_path: ", img_path)
                print("id: ", id)
                gray_img=cv2.resize(test_img,(width,heigth))
                faces.append(gray_img)
                faceID.append(int(id))

    return faces, faceID

def train_classifierLBPH(faces, faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer

def train_classifierEigen(faces, faceID):
    face_recognizer=cv2.face.EigenFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer
