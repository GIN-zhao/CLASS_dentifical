
import sys

from PyQt5 import uic
from functools import partial
import numpy as np
from PyQt5 import *

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
from prompt_toolkit import Application
from main_detect import detect


class Main_Window():
    def __init__(self):

        self.main_window = uic.loadUi('./ui/main.ui')
        self.main_window.BTN2.clicked.connect(self.loadvideo)
        self.video_path = 0
        self.video = cv2.VideoCapture()
        self.timer_camera = QTimer()
        self.slot_init()
        self.main_window.show()

    def loadvideo(self):
        self.video_path = self.main_window.FILE_dir.text().strip()
        print(self.video_path)

    def openCamera(self):
        flag = self.video.open(self.video_path)
        if flag == False:
            msg = QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
                                      buttons=QMessageBox.Ok,
                                      defaultButton=QMessageBox.Ok)
        else:
            self.timer_camera.start(30)
            self.main_window.BTN1.setText('关闭监测')

    def slot_init(self):
        self.timer_camera.timeout.connect(self.show_camera)
        # # 信号和槽连接
        # self.returnButton.clicked.connect(self.returnSignal)
        self.main_window.BTN1.clicked.connect(self.slotCameraButton)

    def slotCameraButton(self):
        if self.timer_camera.isActive() == False:
            print('open')
            # 打开摄像头并显示图像信息
            self.openCamera()
        else:
            # 关闭摄像头并清空显示信息
            self.closeCamera()

    def show_camera(self):
        flag, self.image = self.video.read()
        show = cv2.resize(self.image, (640, 480))
        show = self.process_img(show)
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(
            show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)

        self.main_window.camera_label.setPixmap(QPixmap.fromImage(showImage))

    def closeCamera(self):
        self.timer_camera.stop()
        self.video.release()
        self.main_window.camera_label.clear()
        self.main_window.BTN1.setText('实时监测')

    def process_img(self, img):
        return detect(img)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    test = Main_Window()
    sys.exit(app.exec_())
