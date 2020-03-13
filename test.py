import tensorflow as tf
import time, os, sys, logging, cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from imutils import  url_to_image, opencv2matplotlib
from PIL import Image

from src.tools.face_detectors import RfcnResnet101FaceDetector, SSDMobileNetV1FaceDetector,FasterRCNNFaceDetector, YOLOv2FaceDetector, TinyYOLOFaceDetector
import src.emotion_gender_age_model as ega

vid = cv2.VideoCapture(0)


def transform_images(x_train, size):
    x_train = tf.expand_dims(x_train, 0)
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train

def preprocess_faces(images, image_size=128):
    batch_size = len(images)
    EGA_Input = np.zeros((batch_size, image_size, image_size, 3), dtype=np.float)
    for i, image in enumerate(images):
        _image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) ## Fix this
        rescaled_image = ega.rescale_image(image=_image,input_shape=(image_size,image_size))
        processed_image = np.array(rescaled_image,dtype=float)/255.
        EGA_Input[i] = processed_image
    return EGA_Input

def get_ega(ega_net, face_images):
    ega_input = preprocess_faces(face_images,image_size=128)
    pred_y_e, pred_y_g, pred_y_a = ega_net(ega_input)
    genders = np.argmax(pred_y_g.numpy(),axis=-1)
    emotions = np.argmax(pred_y_e.numpy(),axis=-1)
    ages = np.argmax(pred_y_a.numpy(),axis=-1)
    return emotions, genders, ages



def draw_boxes_labels( images,emotions, genders, ages,color=(0, 0, 255)):
    gender_dict = {0: 'FEMALE', 1: 'MALE'}
    emotion_dict = {0: 'ANGER', 1:'NEUTRAL', 2:'SURPRISE', 3:'HAPPY', 4:'SAD', 5:'FEAR', 6:'DISGUST'}   

MIN_CONFIDENCE=0.5
Detector = SSDMobileNetV1FaceDetector(min_confidence=MIN_CONFIDENCE)
model_path = "src/checkpoints/run4/EGA_epoch_2_score_81.model"
EGA_Net = ega.load_model(model_path)

gender_dict = {0: 'FEMALE', 1: 'MALE'}
emotion_dict = {0: 'ANGER', 1:'NEUTRAL', 2:'SURPRISE', 3:'HAPPY', 4:'SAD', 5:'FEAR', 6:'DISGUST'} 
times = []
while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        t1 = time.time()
        faces_coords, face_images = Detector.detect(img, draw_faces=False)
        if len(faces_coords) > 0:
            emotions, genders, ages = get_ega(EGA_Net,face_images)
            for i, face_image in enumerate(face_images):
                x, y, w, h, confidence = faces_coords[i]
                conf = int(confidence*100)
                color=(50,205,50)
                label = f"conf:{conf},{emotion_dict[emotions[i]]}, {gender_dict[genders[i]]}, {ages[i]} "
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x, y), (x + label_size[0], y + label_size[1] + base_line), color, cv2.FILLED)
                cv2.putText(img, label, (x, y + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


        
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break