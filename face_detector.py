'''
__author__ = 'Song Chae Young'
__date__ = 'Jan.17, 2023'
__email__ = '0.0yeriel@gmail.com'
__fileName__ = 'me_not_me_detector.py'
__github__ = 'SongChaeYoung98'
__status__ = 'Development'
'''


'''
- 이거 바꿔, delete, add 
'''

# Open CV 얼굴 감지기, 메인

import numpy as np

import tensorflow as tf
from tensorflow import keras

import cv2

# Colors to draw rectangles in BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)

VIDEO_FILE = 'datasets/etc/FaceVideo2.mp4'

# opencv object that will detect faces for us
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Model
model_name = 'face_classifier_ResNet152.h5'  # 이거 바꿔

face_classifier = keras.models.load_model(f'models/{model_name}')
class_names = ['me', 'not_me']


def get_extended_image(img, x, y, w, h, k=0.1):
    # The next code block checks that coordinates will be non-negative
    # (in case if desired image is located in top left corner)
    if x - k*w > 0:
        start_x = int(x - k*w)
    else:
        start_x = x
    if y - k*h > 0:
        start_y = int(y - k*h)
    else:
        start_y = y

    end_x = int(x + (1 + k)*w)
    end_y = int(y + (1 + k)*h)

    face_image = img[start_y:end_y,
                     start_x:end_x]
    face_image = tf.image.resize(face_image, [250, 250])
    # shape from (250, 250, 3) to (1, 250, 250, 3)
    face_image = np.expand_dims(face_image, axis=0)
    return face_image


# Streaming
video_capture = cv2.VideoCapture()  # webcamera

if not video_capture.isOpened():
    print("Unable to access the camera, Replaces with saved video files")
    video_capture = cv2.VideoCapture(VIDEO_FILE)  # add
else:
    print("Access to the camera was successfully obtained")
    video_capture = cv2.VideoCapture(0)  # add

print("Streaming started - to quit press ESC")


while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Can't receive frame. Exit Streaming.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        face_image = get_extended_image(frame, x, y, w, h, 0.5)

        result = face_classifier.predict(face_image)  # face_classifier: model
        prediction = class_names[np.array(
            result[0]).argmax(axis=0)]  # predicted class
        confidence = np.array(result[0]).max(axis=0)  # degree of confidence

        if prediction == 'me':
            color = GREEN

        else:
            color = RED



        # draw a rectangle around the face
        cv2.rectangle(frame,        # 원래 'frame' 내가 'blur_face_image'로 변경
                      (x, y),  # start_point
                      (x+w, y+h),  # end_point
                      color,
                      2)  # thickness in px
        cv2.putText(frame,
                    # text to put
                    "{:6} - {:.2f}%".format(prediction, confidence*100),
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN,  # font
                    2,  # fontScale
                    color,
                    2)  # thickness in px









    # display the resulting frame
    cv2.imshow("Real-time Face detector", frame)

    # Exit with ESC
    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC code
        break





video_capture.release()
cv2.destroyAllWindows()