from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure


import matplotlib.pyplot as plt


class MplWidget(QWidget):#MplWidget 을 정의함으로써 Widget을 Designer에서 사용가능

    def __init__(self, parent=None):#상속 x
        QWidget.__init__(self, parent)
        fig = Figure(tight_layout=True)#tight_layout 시 자동으로 간격 조절
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas,self,coordinates=False)
        vertical_layout = QVBoxLayout()#layout에 canvas를 집어 넣기 위함
        vertical_layout.setContentsMargins(0,0,0,0)
        vertical_layout.setSpacing(0)
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget(self.toolbar)
        self.setLayout(vertical_layout)
