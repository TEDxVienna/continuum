import numpy as ny
from time import sleep
import argparse
from PyQt5.QtWidgets import QWidget
from NeuroNetwork import Neuronet
from keras.utils.data_utils import get_file
import Preloader
import os
import time
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from PyQt5.QtGui import QIcon
import sys


class WebcamCV(QtCore.QObject):
    data = QtCore.pyqtSignal(ny.ndarray)

    def __init__(self, cam=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(cam)
        self.timer = QtCore.QBasicTimer()

    def start(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        ret, data = self.camera.read()
        if ret:
            self.data.emit(data)


class FaceAppRender(QtWidgets.QWidget):
    Trained_Model_Path = "\\models\\deploy.hdf5"  # path to pretrained model. You can add your own model here
    Haarcascade_Model_Path = ".\\models\\haarcascade_frontalface_alt.xml"  # path to haarcascade xml

    def __init__(self, depth=16, width=8, face_size=64, parent=None):
        self.face_size = face_size
        self.model = Neuronet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "models").replace("//", "\\")
        fpath = get_file('deploy.hdf5',
                         self.Trained_Model_Path,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)  ##for use with pyqt
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceAppRender, cls).__new__(cls)
        return cls.instance

    @classmethod
    def DrawLabel(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,  # used for drawing labels
                  font_scale=1, thickness=4):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def CropFaces(self, pixelA, section, margin=20, size=40):

        img_height, img_width, _ = pixelA.shape
        if section is None:
            section = [0, 0, img_width, img_height]
        (x, y, width, height) = section
        margin = int(min(width, height) * margin / 100)
        x1 = x - margin
        y1 = y - margin
        x2 = x + width + margin
        y2 = y + height + margin

        if x1 < 0:
            x2 = min(x2 - x1, img_width - 1)
            x1 = 0

        if y1 < 0:
            y2 = min(y2 - y1, img_height - 1)
            y1 = 0

        if x2 > img_width:
            x1 = max(x1 - (x2 - img_width), 0)
            x2 = img_width

        if y2 > img_height:
            y1 = max(y1 - (y2 - img_height), 0)
            y2 = img_height

        cropped = pixelA[y1: y2, x1: x2]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = ny.array(resized_img)
        return resized_img, (x1, y1, x2 - x1, y2 - y1)

    def detect_faces(self, image: ny.ndarray):  #for use with pyqt
        # haarclassifiers work better in black and white
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)

        faces = self.classifier.detectMultiScale(gray_image,
                                                 scaleFactor=1.3,
                                                 minNeighbors=4,
                                                 flags=cv2.CASCADE_SCALE_IMAGE,
                                                 minSize=self._min_size)

        return faces

    def image_data_slot(self, image_data):  #for use with pyqt
        faces = self.detect_faces(image_data)
        for (x, y, w, h) in faces:
            cv2.rectangle(image_data,
                          (x, y),
                          (x+w, y+h),
                          self._red,
                          self._width)

        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: ny.ndarray):    #for use with pyqt
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    @staticmethod
    def __DrawLabel(img, text, pos, bg_color):
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        color = (0, 0, 0)
        thickness = cv2.FILLED
        margin = 2

        txt_size = cv2.getTextSize(text, font_face, scale, thickness)

        end_x = pos[0] + txt_size[0][0] + margin
        end_y = pos[1] - txt_size[0][1] - margin

        cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
        cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


    def DetectFaces(self):
        counter = 1
        red = (0, 0, 255)
        width = 3
        gthreshold = 0.5

        face_cascade = cv2.CascadeClassifier(self.Haarcascade_Model_Path)
        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)


        # infinite loop, break by key ESC
        while True:
            if not video_capture.isOpened():
                sleep(5)
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=4,
                minSize=(self.face_size, self.face_size)
            )
            # placeholder for cropped faces
            face_imgs = ny.empty((len(faces), self.face_size, self.face_size, 3))
            for i, face in enumerate(faces):
                face_img, cropped = self.CropFaces(frame, face, margin=40, size=self.face_size)
                (x, y, w, h) = cropped

                cv2.rectangle(frame,
                              (x, y),
                              (x + w, y + h),
                              red,
                              width)

                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                face_imgs[i, :, :, :] = face_img
            if len(face_imgs) > 0:
                # predict ages and genders of the detected faces
                results = self.model.predict(face_imgs)
                predicted_genders = results[0]
                ages = ny.arange(0, 101).reshape(101, 1)
                appAge = results[1].dot(ages).flatten()
            # draw results
            time.time()
            startTime = time.time()
            label2 = "Please face the camera :)"
           # self.DrawLabel(frame, 200, label2)
            cv2.putText(frame, label2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, red)
            for i, face in enumerate(faces):
                label = "{} year old {}".format(int(appAge[i]), "Woman" if predicted_genders[i][0] > gthreshold else "Man")
                label2 = "Stand still!"
                counter += 1
                self.DrawLabel(frame, (face[0], face[1]), label)
                cv2.putText(frame, label2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        red)
                timeElapsed = startTime + time.time()

                # secElapsed = int(timeElapsed)
                print(counter)

                if counter > 15:
                    #  cv2.putText(frame, "Your predicted age!", (50, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    #             1, (100, 100, 100))
                    cv2.waitKey(0)
                    counter = 0  # reset the counter
                    if cv2.waitKey(5) == 27:
                        break

            cv2.imshow('Continuum', frame)
            if cv2.waitKey(5) == 27:
                break

        #video_capture.release()
        #cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser(description="Continuum",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    depth = args.depth
    width = args.width
    face = FaceAppRender(depth=depth, width=width)
    face.DetectFaces()


if __name__ == "__main__":
    Preloader.main()
