import sys,time
import pandas as pd
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import Class_Transform as CT
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import pywt

from multiprocessing import Process
import nidaqmx
from nidaqmx.constants import AcquisitionType

form_class = uic.loadUiType("TEST.ui")[0] #qt Designer files
ex=uic.loadUiType("Explain.ui")[0]
Setting=uic.loadUiType("Setting.ui")[0]

class Set(QMainWindow,Setting):
    def __init__(self,parent=None):
        super().__init__()
        self.setupUi(self)
        self.Cancel.clicked.connect(self.close)

    def Yes(self):
        global Select,Number_of_Sensor,Number_of_Data,wavelet,Level,Rank
        wavelet = self.MotherWavelet.text()
        Level = self.LevelSpin.value()
        Select = self.SelectSpin.value()
        Number_of_Data=self.Data.value()
        Number_of_Sensor=self.Sensor.value()
        Rank=self.Rank.value()
        self.close()

class Extra(QWidget,ex):
    def __init__(self,parent=None):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("고장진단 학습모델 정보")
        self.Data.setText("Number_of_Data:%d"%Number_of_Data)
        self.Sensor.setText("Number_of_Sensor:%d"%Number_of_Sensor)
        self.Mother.setText("Mother_Wavelet:%s"%wavelet)
        self.Level.setText("WaveletLevel:%d"%Level)
        self.Select.setText("Select_Level:%d"%Select)
        self.Rank.setText("상위 Rank 개수:%d"%Rank)

class MyWindow(QMainWindow, form_class):#qt designer
    def __init__(self):
        super().__init__()#부모Class에서 상속값 가져오기
        self.setupUi(self)#부모객체에서 setupUi 실행 (위젯들 모음)
        #이름
        self.setWindowTitle("고장진단 프로그램")
        #self.setWindowIcon()
        #Push버튼 처리
        self.File_open.clicked.connect(self.Find_Data)
        self.Diagnose.clicked.connect(self.Predict_state)
        self.RawData.clicked.connect(self.RawData_Graph)
        self.Pvaluebutton.clicked.connect(self.P_value_Graph)

        self.Info.clicked.connect(self.New)
        self.Training.clicked.connect(self.Setting)
        self.push.clicked.connect(self.Train)

        #로고 넣기
        picture = QPixmap("./image.PNG")
        scale = picture.scaled(307,281,Qt.KeepAspectRatio)
        self.Image.setPixmap(scale)
        #P-value comboBox에 Item 추가
        self.ADD_RANK()
        self.ADD_Model()
        #위젯디자인
        #Raw Data 위젯
        self.pushButton.clicked.connect(self.Plot)
        #layout에 채널 생성
        NoD = 2
        NoC = 3
        self.checklist = []
        self.selectlist =[]
        for i in range(1,NoD+1):
            self.GLay.addWidget(QLabel("Device_%d:"%i),2*i-1,0)
            for j in range(1,NoC+1):
                a = QCheckBox("Channel_%d"%j)
                b = QComboBox()
                b.addItems(["Acceleration","Voltage","Current"])
                self.checklist.append(a)
                self.selectlist.append(b)
                self.GLay.addWidget(a,2*i,j-1)
                self.GLay2.addWidget(b,(i-1)*NoC+j,1)
                self.GLay2.addWidget(QLabel("D_%d-C_%d"%(i,j)),(i-1)*NoC+j,0)
        for i in range(np.size(self.checklist)):
            self.checklist[i].stateChanged.connect(self.able)


        self.Daq.clicked.connect(self.Act)

    def Act(self):

        plt.ion()
        # sampling time & Hz
        samp = self.Freq.value()
        time = self.Time.value()

        self.MplWidget.canvas.figure.clf()
        ax = self.MplWidget.canvas.figure.add_subplot(111, xlabel='time', ylabel='magnitude', autoscale_on=True,
                                                      title='Raw_Data')

        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan('cDAQ1Mod1/ai0:2')  # daq1의 채널
            task.ai_channels.add_ai_accel_chan('cDAQ1Mod2/ai0:2')  # daq2의 채널
            task.timing.cfg_samp_clk_timing(rate=samp, sample_mode=AcquisitionType.FINITE, samps_per_chan=samp * time)
            # 공간할당
            a = []
            space = np.zeros(shape=(samp * time, task.number_of_channels))
            for i in range(samp * time):
                data = task.read(number_of_samples_per_channel=1)
                if data[5]>3:
                    a.append(data[6])
                else:
                    pass
                ax.scatter(i, data[2], c='r')

            #Reader = nidaqmx.stream_readers.AnalogMultiChannelReader(task.in_stream, auto_start=False)
            #Reader.read_many_sample(space, number_of_samples_per_channel=1)

    def able(self):
        for i in range(np.size(self.checklist)):
            if bool(self.checklist[i].checkState()) == 1:
                self.selectlist[i].setEnabled(True)
            else:
                self.selectlist[i].setEnabled(False)

    def Plot(self):
        text = str(self.Sensor.currentText())
        print(text)
        self.Single.canvas.figure.clf()
        ax = self.Single.canvas.figure.add_subplot(111, xlabel='time', ylabel='magnitude', autoscale_on=True,
                                                   title=text)
        try:
            for j in np.arange(1, self.NoS + 1):
                x = "_%d" % j
                print(x)
                if x in text:
                    ax.plot(self.data[:, j-1], label='Sensor_%d' % j, linewidth=0.4)
                    ax.legend(loc="upper right")
            self.Single.canvas.draw()
        except:
            pass

    def Train(self):
        try:
            b = CT.Training(Number_of_Data,Number_of_Sensor,wavelet,Select,Rank,Level)
            print(Number_of_Data,Number_of_Sensor)
            b.MakeModel(self.pBar)
            self.comp.setText("Complete")
        except:
            QMessageBox.question(self, "오류", "Setting을 먼저 하십시오", QMessageBox.Yes)
    def Setting(self):
        self.train = Set(self)
        self.train.show()
        self.train.Save.clicked.connect(self.train.Yes)
    def New(self):
        try:
            self.new = Extra(self)
            self.new.show()
        except:
            QMessageBox.question(self, "오류", "Setting을 먼저 하십시오", QMessageBox.Yes)
    def ADD_Model(self):
        self.modellist=["PNN_Model","SVM_Model","KNN_Model"]
        self.Model.addItems(self.modellist)
    def ADD_RANK(self):
        self.addlist = []
        for i in np.arange(1,(CT.Rank+1)):
            a="Rank_%d"%i
            self.addlist.append(a)
        self.Select_Rank.addItems(self.addlist)
        self.Select_Rank.insertSeparator(5)
    def RawData_Graph(self):
        text = self.textEdit.toPlainText()
        data2 = pd.read_csv(text)
        self.data = np.array(data2,dtype=np.float32)
        self.NoS = np.size(self.data[0,:])
        list = ["All"]
        for i in np.arange(1,self.NoS+1):
            a="Sensor_%d"%i
            list.append(a)
        self.Sensor.clear()
        self.Sensor.addItems(list)
        self.MplWidget.canvas.figure.clf()
        try:
            t = np.arange(0, 1, 1 / np.size(self.data[:, 0]))
            # self.MplWidget.canvas.ax1.clear()
            ax = self.MplWidget.canvas.figure.add_subplot(111, xlabel='time', ylabel='magnitude', autoscale_on=True,
                                                          title='Raw_Data')
            for j in np.arange(self.NoS):
                ax.plot(self.data[:, j], label='Sensor_%d' % (j + 1), linewidth=0.4)
                ax.legend(loc="upper right")
            # ax2=self.MplWidget.canvas.figure.add_subplot(212,xlabel='time',ylabel='magnitude',autoscale_on=True,title='Sensor 2')
            # ax2.plot(t,data[:, 1])
            self.MplWidget.canvas.draw()
        except:
            QMessageBox.question(self, "오류", "파일을 가져오십시오", QMessageBox.Yes)
    def Find_Data(self):
        fname = QFileDialog.getOpenFileName(self)
        self.textEdit.setText(fname[0])

    def Predict_state(self):
        try:
            mytext = self.textEdit.toPlainText()
            validation = ".csv"
            if validation in mytext:
                a = CT.MakingFeature(Number_of_Data,Number_of_Sensor,wavelet,Select,Rank,Level)#미리 만들어 놓은 Class 호출 후 객체 선언
                self.NewFeature, self.data = a.Transform(mytext)#Rank 에 적합한 Feature와 그래프 그리기 위해 원본 Data 선언
                Mode = str(self.Model.currentText())
                print(Mode)
                with open(Mode+'.pickle', 'rb') as f:
                    Selected_Model = pickle.load(f)
                NewFeature = np.transpose(self.NewFeature)
                Predicted = Selected_Model.predict([NewFeature])
                if Predicted == 1:
                    self.label_2.setText('Normal')
                else:
                    self.label_2.setText('AbNormal')
            else:
                QMessageBox.question(self,"오류","csv파일을 입력하십시오",QMessageBox.Yes)
        except:
            QMessageBox.question(self, "오류", "Setting을 먼저 하십시오", QMessageBox.Yes)
    def P_value_Graph(self):
        try:
            #현재 Combobox 안에 있는 Rank값을 기존 addlist와 비교하여 받아오기
            for i in np.arange(np.size(self.addlist)):
                if self.addlist[i]== str(self.Select_Rank.currentText()):
                    FeatureRank=i
                    break
            k=CT.MakingFeature(Number_of_Data,Number_of_Sensor,wavelet,Select,Rank,Level)
            with open('Select.pickle','rb') as f:#기존에 데이터를 P-value값으로 오름차순 해놓은 Select을 가져옴
                select=pickle.load(f)
            with open('FT.pickle','rb') as f:
                FT=pickle.load(f)
            Normal = FT[0, :, :]
            AbNormal = FT[1, :, :]
            x = Normal[int(select[FeatureRank, 1]), :]
            y = AbNormal[int(select[FeatureRank, 1]), :]
            name = k.Finding(FeatureRank)
            self.P_Value.canvas.figure.clf()#지금까지 있던 figure를 지움--> 겹치지 않게 그리기 위해서
            ax1=self.P_Value.canvas.figure.add_subplot(111,xlabel='Data',ylabel='Density',title=name,label=['Normal','Abnormal','Position'])
            sns.kdeplot(x,ax=ax1,color='blue',label='Normal')
            sns.kdeplot(y,ax=ax1,color='red',label = 'Abnoraml')
            ax1.axvline(x=self.NewFeature[FeatureRank],color='black',label='DATA')#axvline= x에대한 상수함수, axhline = y에대한 상수 함수
            ax1.legend(loc="upper right")
            self.P_Value.canvas.draw()
        except:
            QMessageBox.question(self, "오류", "진단을 먼저 하십시오", QMessageBox.Yes)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

## try except로 오류 만들기
