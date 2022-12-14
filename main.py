import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QTimer
import numpy as np
import cv2
import threading
import csv
from main_ui import *
import gxipy as gx
from OpenGL.GL import *
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
import scipy.io as scio


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

lines = [
    [11,12],
    [12,14],
    [14,16],
    [16,22],
    [16,20],
    [16,18],
    [12,24],
    [24,26],
    [26,28],
    [28,32],
    [28,30],
    [32,30],
    [11,13],
    [13,15],
    [15,17],
    [15,19],
    [15,21],
    [11,23],
    [23,24],
    [23,25],
    [25,27],
    [27,29],
    [27,31],
    [29,31]
]

keyPointDef = {
    "鼻":0,
    "左眼":7,
    "右眼":8,
    "左肩":11,
    "右肩":12,
    "左侧肘关节":13,
    "右侧肘关节":14,
    "左手腕":15,
    "右手腕":16,
    "左手-小指":17,
    "右手-小指":18,
    "左手-食指":19,
    "右手-食指":20,
    "左手-拇指":21,
    "右手-拇指":22,
    "左侧髋关节":23,
    "右侧髋关节":24,
    "左侧膝关节":25,
    "右侧膝关节":26,
    "左侧脚踝":27,
    "右侧脚踝":28,
    "左侧脚跟":29,
    "右侧脚跟":30,
    "左侧脚尖":31,
    "右侧脚尖":32

}

cameraScale = 1.0
dt = 0.01

class MainFrame(QMainWindow, Ui_MainWindow):
    def __init__(self) -> None:
        
        # basic configuration
        self.app = QApplication(sys.argv)
        self.app.aboutToQuit.connect(self.exitHandler)
        super().__init__()
        self.widget = QWidget(parent=self)
        self.ui = Ui_MainWindow()
        
        self.ui.setupUi(self)
        self.bind_func()
        self.cam_open = False
        self.file_open = False
        self.inputSource = None
        self.timer = QTimer(self)
        self.th_dataplot = threading.Thread(target=self.openglCurveShow)
        self.th_stickplot = threading.Thread(target=self.openglStickShow)
        self.timer_stick = QTimer(self)
        self.pause = False

        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.playing = False
        self.cord_x = [[0.] for i in range(33)]
        self.cord_y = [[0.] for i in range(33)]
        self.cord_z = [[0.] for i in range(33)]
        self.v_x = [[] for i in range(33)]
        self.v_y = [[] for i in range(33)]
        self.v_z = [[] for i in range(33)]
        self.acc = [[] for i in range(33)]
        self.frame = 0
        self.results = None
        self.keypoint = 0
        self.notExported = False
        self.z_angle = 30
        self.keepRunning = True
    
    def resizeEvent(self, event):
        #print("entering this event.")
        width = self.width()
        height = self.height()
        self.ui.horizontalFrame.setGeometry(QtCore.QRect(50, 40, int(width * 1321/1415), int(height * 701/800)))
        

    def exitHandler(self):
        self.keepRunning = False

    def startup(self):
        self.initPlotSettings()
        self.statusBar().showMessage("ready")
        self.ui.comboBox.addItems(list(keyPointDef.keys()))
        self.ui.comboBox.setCurrentIndex(0)
        pass
        
    def bind_func(self):
        self.ui.pushButton_5.clicked.connect(self.set_camera)
        self.ui.pushButton_6.clicked.connect(self.import_video_file)
        self.ui.actionselect_video.setShortcut("Ctrl+I")
        self.ui.actionselect_video.triggered.connect(self.import_video_file)
        self.ui.pushButton_2.clicked.connect(self.btnPlay)
        self.ui.pushButton.clicked.connect(self.btnStop)
        self.ui.pushButton_3.clicked.connect(self.btnPause)
        self.ui.pushButton_4.clicked.connect(self.resetProg)
        self.ui.comboBox.currentIndexChanged.connect(self.selectKeypoint)
        self.ui.actioncsv.triggered.connect(self.exportCsv)
        self.ui.actiontxt.triggered.connect(self.exportTxt)
        self.ui.actionmat.triggered.connect(self.exportMat)
        self.ui.actionnpy.triggered.connect(self.exportNpy)

    def init_camera(self):
        self.device_manager = gx.DeviceManager()
        self.dev_num, dev_info_list = self.device_manager.update_device_list()

        # 获取设备基本信息列表
        strSN = dev_info_list[0].get("sn")
        # 通过序列号打开设备
        if self.dev_num == 0:
            #print("Number of enumerated devices is 0")
            QMessageBox.critical(self.widget, "错误警告", "当前无可用摄像机设备，请检查连接情况！")
            
            self.cam_open = False
            return False
        return True
        
    def selectKeypoint(self):
        self.keypoint = keyPointDef.get(self.ui.comboBox.currentText())
    
    def set_camera(self):
        self.resetProg()
        if not self.init_camera():
            return
        
        devices = ['%d'%i for i in range(1, self.dev_num + 1)]
        value, confirmed = QInputDialog.getItem(self,"选择设备号","请选择需要打开的设备型号", devices, 0, False)
        value = int(value)
        if not confirmed:
            return
        
        if (self.cam_open and self.cam_number == int(value)):
            return
        try:
            self.cam = self.device_manager.open_device_by_index(int(value))
            
        except:
            print("Cannot open camera with index %d"%int(value))
            pass
        self.cam_number = int(value)
        self.file_open = False
        self.cam_open = True
        self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
        self.cam.ExposureTime.set(20000.0)
        self.cam.stream_on()

    def import_video_file(self):
        self.resetProg()
        fname, ftype = QFileDialog.getOpenFileName(self, "选择视频", "./", "(*.mp4);;(*.avi)")
        print(fname)
        if fname is None or len(fname) < 1:
            QMessageBox.critical(self, "未选取文件", "没有选择正确的视频文件")
            self.file_open = False
            return
        self.file_open = True
        self.inputSource = cv2.VideoCapture(fname)

    
    def openglCamShow(self):
        t_begin = time.time()
        if not self.cam_open:
            #self.ui.openGLWidget.loadImg(None)
            return
        img = self.cam.data_stream[0].get_image()
        if img is None:
            return
        print("timecost:",time.time() - t_begin)
        img = img.convert('RGB').get_numpy_array()
        
        

        img = cv2.resize(img, (self.ui.openGLWidget.width(),self.ui.openGLWidget.height()))
        results = self.pose.process(img)
        mp_drawing.draw_landmarks(
        img,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
        self.results = results
        rows, cols, _ = img.shape

        if results.pose_world_landmarks is None:
            pass
        else:
            for index, points in enumerate(results.pose_world_landmarks.landmark):
                self.cord_x[index].append((1 - points.x) * cols * cameraScale)
                self.cord_y[index].append(points.z * cols * cameraScale)
                self.cord_z[index].append((1 - points.y) * rows * cameraScale)
                self.v_x[index].append((self.cord_x[index][self.frame + 1] - self.cord_x[index][self.frame])/dt)
                self.v_y[index].append((self.cord_y[index][self.frame + 1] - self.cord_y[index][self.frame])/dt)
                self.v_z[index].append((self.cord_z[index][self.frame + 1] - self.cord_z[index][self.frame])/dt)
                self.acc[index].append(np.linalg.norm([self.v_x[index][self.frame], self.v_y[index][self.frame], self.v_z[index][self.frame]], ord= 2) / dt)
            
            self.frame += 1

        height, width, _ = img.shape
        
        Qimg = QImage(img.data, width, height, 3 * width, QImage.Format_RGB888)
        self.ui.openGLWidget.loadImg(Qimg)
        self.ui.openGLWidget.update()
        
    def openglVideoShow(self):
        if not self.file_open:
            return
        reg, img = self.inputSource.read()
        if not reg:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.ui.openGLWidget.width(),self.ui.openGLWidget.height()))
        results = self.pose.process(img)
        mp_drawing.draw_landmarks(
        img,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
        self.results = results
        rows, cols, _ = img.shape
        if results.pose_world_landmarks is None:
            pass
        else:
            for index, points in enumerate(results.pose_world_landmarks.landmark):
                self.cord_x[index].append((1 - points.x) * cols * cameraScale)
                self.cord_y[index].append(points.z * cols * cameraScale)
                self.cord_z[index].append((1 - points.y) * rows * cameraScale)
                self.v_x[index].append((self.cord_x[index][self.frame + 1] - self.cord_x[index][self.frame])/dt)
                self.v_y[index].append((self.cord_y[index][self.frame + 1] - self.cord_y[index][self.frame])/dt)
                self.v_z[index].append((self.cord_z[index][self.frame + 1] - self.cord_z[index][self.frame])/dt)
                self.acc[index].append(np.linalg.norm([self.v_x[index][self.frame], self.v_y[index][self.frame], self.v_z[index][self.frame]], ord= 2) / dt)

            self.frame += 1
        height, width, _ = img.shape
        Qimg = QImage(img.data, width, height, 3 * width, QImage.Format_RGB888)
        self.ui.openGLWidget.loadImg(Qimg)
        self.ui.openGLWidget.update()
    
        
    def initPlotSettings(self):
        # plot 1
        self.dataFig = plt.figure()
        dataFig = self.dataFig
        self.vxplt = dataFig.add_subplot(4,1,1)
        self.vyplt = dataFig.add_subplot(4,1,2)
        self.vzplt = dataFig.add_subplot(4,1,3)
        self.accplt = dataFig.add_subplot(4,1,4)

        # plot stick
        self.fig = plt.figure(dpi = 50)
        self.ax = self.fig.add_subplot(111, projection='3d')


        
    def openglCurveShow(self):
        while self.keepRunning:
            frame = self.frame

            begin = max(1, frame - 50)
            keypoint = self.keypoint
            v_x = self.v_x
            v_y = self.v_y
            v_z = self.v_z
            acc = self.acc
            self.vxplt.plot(v_x[keypoint][begin:frame + 1]);self.vxplt.set_ylabel("v_x");self.vxplt.get_xaxis().set_visible(False)
            self.vyplt.plot(v_y[keypoint][begin:frame + 1]);self.vyplt.set_ylabel("v_y");self.vyplt.get_xaxis().set_visible(False)
            self.vzplt.plot(v_z[keypoint][begin: frame + 1]);self.vzplt.set_ylabel("v_z");self.vzplt.get_xaxis().set_visible(False)
            self.accplt.plot(acc[keypoint][begin: frame + 1]);self.accplt.set_ylabel("acc"); self.accplt.set_xlabel("t")

            canvas = FigCanvas(self.dataFig)
            canvas.draw()
            img = np.array(canvas.get_renderer().buffer_rgba())
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            img = cv2.resize(img, (self.ui.openGLWidget_2.width(),self.ui.openGLWidget_2.height()))
            #print("curve show time cost is", time.time() - tbegin)
            height, width, _ = img.shape
            #print("len:",len(v_x[keypoint]))
            Qimg = QImage(img.data, width, height, 3 * width, QImage.Format_RGB888)
            self.ui.openGLWidget_2.loadImg(Qimg)
            self.ui.openGLWidget_2.update()
            self.vxplt.clear()
            self.vyplt.clear()
            self.vzplt.clear()
            self.accplt.clear()

            time.sleep(0.01)
        #pass

    def openglStickShow(self):
        
        #landmarks = []
        if self.results is None:
            return
        if self.results.pose_world_landmarks is None:
            return
        cols, rows = self.ui.openGLWidget.width(), self.ui.openGLWidget.height()
        allpoint = []
        cols, rows = 0.6, 0.6
        for index, points in enumerate(self.results.pose_world_landmarks.landmark):
            #landmarks.append(points)
            allpoint.append([-points.x * cols * 0.7, points.z * cols , -points.y * rows * 1.1 + 0.1 ])
        omitPoints = range(1, 11)

        self.ui.openGLWidget_3.loadMat(allpoint, lines, self.z_angle, omitPoints)
        self.ui.openGLWidget_3.update()        
        
        if self.ui.radioButton.isChecked():
            self.z_angle += 1

    def resetProg(self):
        if self.notExported:
            reset = QMessageBox().question(self, "保存提醒","你的数据尚未保存，是否确定要初始化？")
            if reset == QMessageBox.No:
                return
        self.btnStop()
        self.cord_x = [[0.] for i in range(33)]
        self.cord_y = [[0.] for i in range(33)]
        self.cord_z = [[0.] for i in range(33)]
        self.v_x = [[] for i in range(33)]
        self.v_y = [[] for i in range(33)]
        self.v_z = [[] for i in range(33)]
        self.acc = [[] for i in range(33)]
        self.frame = 0
        self.notExported = False 
    
    def btnPlay(self):
        if not self.playing:
            if not self.cam_open  and not self.file_open:
                QMessageBox().warning(self, "播放警告","未选择播放源")
                return
            if self.pause:
                self.pause = False
            elif self.cam_open:
                self.notExported = True
                self.timer.timeout.connect(self.openglCamShow)
            else:
                self.notExported = True
                self.timer.timeout.connect(self.openglVideoShow)
            
            self.timer_stick.timeout.connect(self.openglStickShow)
            self.timer_stick.start(10)
            if not self.th_dataplot.isAlive():
                self.th_dataplot.start()
            self.timer.start(10)
            self.playing = True
    
    def btnPause(self):
        if self.playing:
            self.timer.stop() 
            self.timer_stick.stop()   
            self.pause = True   
            self.playing = False
    
    def btnStop(self):
        self.pause = False
        if self.playing:
            self.timer.stop()
            self.timer_stick.stop()
            self.playing = False
        if self.cam_open:
            self.cam.stream_off()
            self.cam.close_device()
            self.cam_open = False
        if self.file_open:
            self.file_open = False
            self.inputSource.release()
            self.inputSource = None

            
    def selectFilePass(self):
        if not self.notExported:
            QMessageBox().warning(self, "操作不可用", "没有可以导出的数据")
            return None
        self.notExported = False
        return QtWidgets.QFileDialog.getExistingDirectory(parent=self)
    def exportCsv(self):
        path = self.selectFilePass()
        
        if path is None:
            return
        frame = self.frame
        v_x = self.v_x
        v_y = self.v_y
        v_z = self.v_z
        acc = self.acc

        with open(path + "/v_x.csv",'w+') as vxf:
            vxf.truncate()
            writer = csv.writer(vxf)
            writer.writerows(v_x)
            vxf.close()

        with open(path + "/v_y.csv",'w+') as vxf:
            vxf.truncate()
            writer = csv.writer(vxf)
            writer.writerows(v_y)
            vxf.close()

        with open(path + "/v_z.csv",'w+') as vxf:
            vxf.truncate()
            writer = csv.writer(vxf)
            writer.writerows(v_z)
            vxf.close()

        with open(path + "/acc.csv",'w+') as vxf:
            vxf.truncate()
            writer = csv.writer(vxf)
            writer.writerows(acc)
            vxf.close()

        QMessageBox().information(self,"文件已导出","已生成4个csv文件")
        self.continueSaving()

        
    def continueSaving(self):
        msb = QMessageBox().question(self,"数据保存提醒","是否需要保留工作区的数据以备后续导出？", QMessageBox.Yes|QMessageBox.No)
        if msb == QMessageBox.Yes:
            self.notExported = True
    def exportTxt(self):
        
        
        path = self.selectFilePass()
        
        if path is None:
            return
        frame = self.frame
        v_x = self.v_x
        v_y = self.v_y
        v_z = self.v_z
        acc = self.acc

        with open(path + "/v_x.txt",'w+') as vxf:
            vxf.truncate()
            for index in range(1, frame):
                for pnt in range(33):
                    vxf.write("%f "%v_x[pnt][index])
                vxf.write('\n\n')
            vxf.close()

        with open(path + "/v_y.txt",'w+') as vxf:
            vxf.truncate()
            for index in range(1, frame):
                for pnt in range(33):
                    vxf.write("%f "%v_y[pnt][index])
                vxf.write('\n\n')
            vxf.close()

        with open(path + "/v_z.txt",'w+') as vxf:
            vxf.truncate()
            for index in range(1, frame):
                for pnt in range(33):
                    vxf.write("%f "%v_z[pnt][index])
                vxf.write('\n\n')
            vxf.close()

        with open(path + "/acc.txt",'w+') as vxf:
            vxf.truncate()
            for index in range(1, frame):
                for pnt in range(33):
                    vxf.write("%f "%acc[pnt][index])
                vxf.write('\n\n')
            vxf.close()

        QMessageBox().information(self,"文件已导出","已生成4个txt文件")
        self.continueSaving()
        
    def exportNpy(self):
        path = self.selectFilePass()
        
        if path is None:
            return
        frame = self.frame
        v_x = np.array(self.v_x)
        v_y = np.array(self.v_y)
        v_z = np.array(self.v_z)
        acc = np.array(self.acc)

        np.save(path + '/v_x.npy', v_x)
        np.save(path + '/v_y.npy', v_y)
        np.save(path + '/v_z.npy', v_z)
        np.save(path + '/acc.npy', acc)
        QMessageBox().information(self,"文件已导出","已生成4个npy文件")
        self.continueSaving()

    def exportMat(self):
        path = self.selectFilePass()
        
        if path is None:
            return
        frame = self.frame
        v_x = np.array(self.v_x)
        v_y = np.array(self.v_y)
        v_z = np.array(self.v_z)
        acc = np.array(self.acc)
        cord_x = np.array(self.cord_x)
        cord_y = np.array(self.cord_y)
        cord_z = np.array(self.cord_z)

        scio.savemat(path + '/v_x.mat', {'v_x%d'%i : v_x[i] for i in range(33)})
        scio.savemat(path + '/v_y.mat', {'v_y%d'%i : v_y[i] for i in range(33)})
        scio.savemat(path + '/v_z.mat', {'v_z%d'%i : v_z[i] for i in range(33)})
        scio.savemat(path + '/acc.mat', {'acc%d'%i : acc[i] for i in range(33)})
        scio.savemat(path + '/cord_x.mat', {'cord_x%d'%i : cord_x[i] for i in range(33)})
        scio.savemat(path + '/cord_y.mat', {'cord_y%d'%i : cord_y[i] for i in range(33)})
        scio.savemat(path + '/cord_z.mat', {'cord_z%d'%i : cord_z[i] for i in range(33)})
        QMessageBox().information(self,"文件已导出","已生成7个mat文件")
        self.continueSaving()
    

    def run(self):
        self.startup()
        self.show()
        sys.exit(self.app.exec())


if __name__ == '__main__':
    MainFrame().run()