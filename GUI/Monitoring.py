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

form_class = uic.loadUiType("Monitoring.ui")[0] # qt Designer파일 불러오기

class Thread(QThread):
    signal = pyqtSignal(name = "tic")

    def __init__(self):
        super().__init__()
        self.mutex = QMutex()
        self.cop = QWaitCondition()
        self._status = False
    def run(self):

        while(1):
            self.mutex.lock()
            if not self._status:
                self.cop.wait(self.mutex)
            else:
                self.signal.emit()
            self.mutex.unlock()
    def toggle(self):
        self._status = not self._status
        if self._status:
            self.cop.wakeAll()
    @property
    def status(self):
        return self._status

class MyWindow(QMainWindow, form_class):#불러온 qt designer와 qmainwindow 상속
    def __init__(self):
        super().__init__()#부모Class에서 상속값 가져오기
        self.setupUi(self)#부모객체에서 setupUi 실행 (위젯들 모음)
        #이름
        self.setWindowTitle("고장진단 프로그램")
        #self.setWindowIcon()
        #Push버튼 처리
        self.Pvaluebutton.clicked.connect(self.P_value_Graph)

        self.sig = Thread()
        self.sig.start()
        self.Daq.clicked.connect(self.test)
        self.sig.signal.connect(self.Act)


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

    def able(self):
        for i in range(np.size(self.checklist)):
            if bool(self.checklist[i].checkState()) == 1:
                self.selectlist[i].setEnabled(True)
            else:
                self.selectlist[i].setEnabled(False)

        self.Daq.clicked.connect(self.Act)
    def test(self):
        self.sig.toggle()
        a = {True:"do",False:"not"}.get(self.sig.status)
        self.Daq.setText(a)


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

    def ADD_RANK(self):
        self.addlist = []
        for i in np.arange(1,(CT.Rank+1)):
            a="Rank_%d"%i
            self.addlist.append(a)
        self.Select_Rank.addItems(self.addlist)
        self.Select_Rank.insertSeparator(5)
    def P_value_Graph(self):
        try:
            # 현재 Combobox 안에 있는 Rank값을 기존 addlist와 비교하여 받아오기
            for i in np.arange(np.size(self.addlist)):
                if self.addlist[i]== str(self.Select_Rank.currentText()):
                    FeatureRank=i
                    break
            k=CT.MakingFeature(Number_of_Data,Number_of_Sensor,wavelet,Select,Rank,Level)
            with open('Select.pickle','rb') as f: # 기존에 데이터를 P-value값으로 오름차순 해놓은 Select을 가져옴
                select=pickle.load(f)
            with open('FT.pickle','rb') as f:
                FT=pickle.load(f)
            Normal = FT[0, :, :]
            AbNormal = FT[1, :, :]
            x = Normal[int(select[FeatureRank, 1]), :]
            y = AbNormal[int(select[FeatureRank, 1]), :]
            name = k.Finding(FeatureRank)
            self.P_Value.canvas.figure.clf() # 지금까지 있던 figure를 지움--> 겹치지 않게 그리기 위해서
            ax1=self.P_Value.canvas.figure.add_subplot(111,xlabel='Data',ylabel='Density',title=name,label=['Normal','Abnormal','Position'])
            sns.kdeplot(x,ax=ax1,color='blue',label='Normal')
            sns.kdeplot(y,ax=ax1,color='red',label = 'Abnoraml')
            ax1.axvline(x=self.NewFeature[FeatureRank],color='black',label='DATA') # axvline= x에대한 상수함수, axhline = y에대한 상수 함수
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