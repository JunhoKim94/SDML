# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:22:21 2019

@author: junho
"""
import numpy as np
import pandas as pd
import scipy.stats as sp
import pywt
import pickle
from neupy import algorithms
from sklearn import svm,metrics
from sklearn.neighbors import KNeighborsClassifier
import time
#INPUT
Number_of_Data=100
Number_of_Sensor=2
Number_of_Feature=9


Level=12
select=9

Rank = 10

class MakingFeature():
    def __init__(self,Number_of_Data,Number_of_Sensor,MotherWavelet,Select,Rank,Level):
        self.Number_of_Data = Number_of_Data
        self.Number_of_Sensor=Number_of_Sensor
        self.MotherWavelet = MotherWavelet
        self.Select = Select
        self.Rank = Rank
        self.Level = Level
        self.Number_of_Feature=9
    def rms(self,y): #rms 함수 정의
        return np.sqrt(np.mean(y**2)) 
    #Time Domain Feature 추출    
    def Transform(self,x):
        TD_Normal = np.zeros(shape=(self.Number_of_Sensor*self.Number_of_Feature))
        path2 = x
        data2 = pd.read_csv(path2)
        a=np.array(data2,dtype=np.float32)
        for j in np.arange(0,self.Number_of_Sensor):
            TD_Normal[self.Number_of_Feature*j+0]=np.max(a[:,j])
            TD_Normal[self.Number_of_Feature*j+1]=np.min(a[:,j])
            TD_Normal[self.Number_of_Feature*j+2]=np.mean(a[:,j])
            TD_Normal[self.Number_of_Feature*j+3]=np.var(a[:,j])
            TD_Normal[self.Number_of_Feature*j+4]=self.rms(a[:,j])
            TD_Normal[self.Number_of_Feature*j+5]=sp.skew(a[:,j])
            TD_Normal[self.Number_of_Feature*j+6]=sp.kurtosis(a[:,j])
            TD_Normal[self.Number_of_Feature*j+7]=self.rms(a[:,j])/np.mean(a[:,j])
            TD_Normal[self.Number_of_Feature*j+8]=np.max(a[:,j])/np.mean(a[:,j])
            #Normal[10*j+9,j]=rms(a[:,j])    
    #Frequency Domain Feature 추출
        FD_Normal = np.zeros(shape=(self.Number_of_Sensor*self.Number_of_Feature*self.Select))
        path2 = x
        data2 = pd.read_csv(path2)
        b=np.array(data2,dtype=np.float32)
        # Mother Wavelet 및 level 설정
        wavelet = pywt.Wavelet(self.MotherWavelet)
        Coef2=pywt.wavedec(b,wavelet,level=self.Level,axis=0)#Wavelet Transform Coefficient
        for j in np.arange(self.Number_of_Sensor):
            for k in np.arange(self.Select):
                coef2=Coef2[self.Level-k]
                FD_Normal[self.Number_of_Feature*j*self.Select+k*self.Number_of_Feature+0]=np.max(coef2[:,j])
                FD_Normal[self.Number_of_Feature*j*self.Select+k*self.Number_of_Feature+1]=np.min(coef2[:,j])
                FD_Normal[self.Number_of_Feature*j*self.Select+k*self.Number_of_Feature+2]=np.mean(coef2[:,j])
                FD_Normal[self.Number_of_Feature*j*self.Select+k*self.Number_of_Feature+3]=np.var(coef2[:,j])
                FD_Normal[self.Number_of_Feature*j*self.Select+k*self.Number_of_Feature+4]=self.rms(coef2[:,j])
                FD_Normal[self.Number_of_Feature*j*self.Select+k*self.Number_of_Feature+5]=sp.skew(coef2[:,j])
                FD_Normal[self.Number_of_Feature*j*self.Select+k*self.Number_of_Feature+6]=sp.kurtosis(coef2[:,j])
                FD_Normal[self.Number_of_Feature*j*self.Select+k*self.Number_of_Feature+7]=self.rms(coef2[:,j])/np.mean(coef2[:,j])
                FD_Normal[self.Number_of_Feature*j*self.Select+k*self.Number_of_Feature+8]=np.max(coef2[:,j])/np.mean(coef2[:,j])
    
        FT=np.concatenate((TD_Normal,FD_Normal),axis=0)
    #Time Domain과 Frequency Domain 특성값 합치기
    
    #p-value 기반 상위 특징요소 선택하기
        with open('./ref/Select.pickle','rb') as f:#기존에 데이터를 P-value값으로 오름차순 해놓은 Select을 가져옴
            Select=pickle.load(f)
        New_Feature=np.zeros(shape=(self.Rank))
        for i in np.arange(self.Rank):
            New_Feature[i]=np.array(FT[int(Select[i,1])])
        return New_Feature,a
    def Finding(self, x):
        with open('./ref/Select.pickle','rb') as f:#기존에 데이터를 P-value값으로 오름차순 해놓은 Select을 가져옴
            self.Select=pickle.load(f)
        Feature_Order = self.Select[x, 1]
        a=Feature_Order%self.Number_of_Feature
        Feature={0:'Max',1:'Min',2:'Mean',3:'Var',4:'RMS',5:'Skewness',6:'Kurtosis',7:'Crest factor',8:'Impulse factor'}.get(a,'deafault')#a값을 대입 후 나머지 deafault
        b1 = Feature_Order%(self.Number_of_Sensor*self.Number_of_Feature)
        b = b1//self.Number_of_Feature
        Sensor='Sensor_%d'%(b+1)
        c=Feature_Order//(self.Number_of_Feature*self.Number_of_Sensor)
        Level={0:'Time_Domain',1:'Approximate'}.get(c,'D_%d Wavelet'%c)
        return 'Feature=%s, Sensor=%s, %s'%(Feature,Sensor,Level)
#######################################################################################################################################
class Training():
    def __init__(self,Number_of_Data,Number_of_Sensor,MotherWavelet,Select,Rank,Level):
        self.Number_of_Data = Number_of_Data
        self.Number_of_Sensor=Number_of_Sensor
        self.MotherWavelet = MotherWavelet
        self.Select = Select
        self.Rank = Rank
        self.Level = Level
        self.Number_of_Feature=9

    def rms(self,y): #rms 함수 정의
        return np.sqrt(np.mean(y**2))

    def MakeModel(self,pBar):
        TD_Abnormal = np.zeros(shape=(self.Number_of_Sensor * self.Number_of_Feature, self.Number_of_Data))
        TD_Normal = np.zeros(shape=(self.Number_of_Sensor * self.Number_of_Feature, self.Number_of_Data))
        # Time Domain Feature 추출
        for i in np.arange(1, self.Number_of_Data + 1):
            path1 = './Normal/Normal_%d.csv' % i
            data = pd.read_csv(path1)
            a = np.array(data, dtype=np.float32)
            path2 = './Abnormal/Abnormal_%d.csv' % i
            data2 = pd.read_csv(path2)
            b = np.array(data2, dtype=np.float32)
            for j in np.arange(0, self.Number_of_Sensor):
                TD_Normal[self.Number_of_Feature * j + 0, i - 1] = np.max(a[:, j])
                TD_Normal[self.Number_of_Feature * j + 1, i - 1] = np.min(a[:, j])
                TD_Normal[self.Number_of_Feature * j + 2, i - 1] = np.mean(a[:, j])
                TD_Normal[self.Number_of_Feature * j + 3, i - 1] = np.var(a[:, j])
                TD_Normal[self.Number_of_Feature * j + 4, i - 1] = self.rms(a[:, j])
                TD_Normal[self.Number_of_Feature * j + 5, i - 1] = sp.skew(a[:, j])
                TD_Normal[self.Number_of_Feature * j + 6, i - 1] = sp.kurtosis(a[:, j])
                TD_Normal[self.Number_of_Feature * j + 7, i - 1] = self.rms(a[:, j]) / np.mean(a[:, j])
                TD_Normal[self.Number_of_Feature * j + 8, i - 1] = np.max(a[:, j]) / np.mean(a[:, j])
                # Normal[10*j+9,j]=rms(a[:,j])
                TD_Abnormal[self.Number_of_Feature * j + 0, i - 1] = np.max(b[:, j])
                TD_Abnormal[self.Number_of_Feature * j + 1, i - 1] = np.min(b[:, j])
                TD_Abnormal[self.Number_of_Feature * j + 2, i - 1] = np.mean(b[:, j])
                TD_Abnormal[self.Number_of_Feature * j + 3, i - 1] = np.var(b[:, j])
                TD_Abnormal[self.Number_of_Feature * j + 4, i - 1] = self.rms(b[:, j])
                TD_Abnormal[self.Number_of_Feature * j + 5, i - 1] = sp.skew(b[:, j])
                TD_Abnormal[self.Number_of_Feature * j + 6, i - 1] = sp.kurtosis(b[:, j])
                TD_Abnormal[self.Number_of_Feature * j + 7, i - 1] = self.rms(b[:, j]) / np.mean(b[:, j])
                TD_Abnormal[self.Number_of_Feature * j + 8, i - 1] = np.max(b[:, j]) / np.mean(b[:, j])
                # Normal[10*j+9,j]=rms(a[:,j])
            #진행상황1
        delay=i/Number_of_Data*40
        pBar.setValue(delay)
        time.sleep(0.005)
        # Time Domain 의 Normal 과 Abnormal에 대한 공간 할당 + 선언--> np.array사용은 필요없음
        T = np.array([TD_Normal, TD_Abnormal])

        # Mother Wavelet 및 level 설정

        wavelet = pywt.Wavelet(self.MotherWavelet)
        # Frequency Domain Feature 추출

        FD_Abnormal = np.zeros(shape=(self.Number_of_Sensor * self.Number_of_Feature * self.Select, self.Number_of_Data))
        FD_Normal = np.zeros(shape=(self.Number_of_Sensor * self.Number_of_Feature * self.Select, self.Number_of_Data))
        for i in np.arange(1, self.Number_of_Data + 1):
            path1 = './Normal/Normal_%d.csv' % i
            data = pd.read_csv(path1)
            a = np.array(data, dtype=np.float32)
            Coef = pywt.wavedec(a, wavelet, level=self.Level, axis=0)  # Wavelet Transform Coefficient
            path2 = './Abnormal/Abnormal_%d.csv' % i
            data2 = pd.read_csv(path2)
            b = np.array(data2, dtype=np.float32)
            Coef2 = pywt.wavedec(b, wavelet, level=self.Level, axis=0)  # Wavelet Transform Coefficient
            for j in np.arange(self.Number_of_Sensor):
                for k in np.arange(self.Select):
                    coef = Coef[self.Level - k]
                    coef2 = Coef2[self.Level - k]
                    FD_Normal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 0, (i - 1)] = np.max(coef[:, j])
                    FD_Normal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 1, (i - 1)] = np.min(coef[:, j])
                    FD_Normal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 2, (i - 1)] = np.mean(coef[:, j])
                    FD_Normal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 3, (i - 1)] = np.var(coef[:, j])
                    FD_Normal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 4, (i - 1)] = self.rms(coef[:, j])
                    FD_Normal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 5, (i - 1)] = sp.skew(coef[:, j])
                    FD_Normal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 6, (i - 1)] = sp.kurtosis(coef[:, j])
                    FD_Normal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 7, (i - 1)] = self.rms(
                        coef[:, j]) / np.mean(coef[:, j])
                    FD_Normal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 8, (i - 1)] = np.max(
                        coef[:, j]) / np.mean(coef[:, j])

                    FD_Abnormal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 0, (i - 1)] = np.max(coef2[:, j])
                    FD_Abnormal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 1, (i - 1)] = np.min(coef2[:, j])
                    FD_Abnormal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 2, (i - 1)] = np.mean(coef2[:, j])
                    FD_Abnormal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 3, (i - 1)] = np.var(coef2[:, j])
                    FD_Abnormal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 4, (i - 1)] = self.rms(coef2[:, j])
                    FD_Abnormal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 5, (i - 1)] = sp.skew(coef2[:, j])
                    FD_Abnormal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 6, (i - 1)] = sp.kurtosis(
                        coef2[:, j])
                    FD_Abnormal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 7, (i - 1)] = self.rms(
                        coef2[:, j]) / np.mean(coef2[:, j])
                    FD_Abnormal[self.Number_of_Feature * j * self.Select + k * self.Number_of_Feature + 8, (i - 1)] = np.max(
                        coef2[:, j]) / np.mean(coef2[:, j])
            #진행상황2
            delay=delay+(i / Number_of_Data * 40)
            pBar.setValue(delay)
            time.sleep(0.005)

        F = np.array([FD_Normal, FD_Abnormal])

        FT = np.concatenate((T, F), axis=1)  # Time Domain과 Frequency Domain 특성값 합치기

        # 데이터 저장 불러오기

        with open('./ref/FT.pickle', 'wb') as f:
            pickle.dump(FT, f, pickle.HIGHEST_PROTOCOL)

        b = np.zeros(shape=(self.Number_of_Sensor * self.Number_of_Feature * (self.Select + 1), 2))
        # size(matrix,n)--> matrix의 n번째 차원의 size를 구하라
        for i in np.arange(self.Number_of_Sensor * self.Number_of_Feature * (self.Select + 1)):
            a = np.array(sp.ttest_ind(FT[1, i, :], FT[0, i, :], equal_var=False))
            # sp.ttest_ind 두개의 독립표본 t test--> 두 집단 사이의 Mean 비교
            # 고장과 정상에서의 Feature들이 변인이 되는가?
            b[i, 0] = a[1]
            b[i, 1] = i
        b = pd.DataFrame(b)
        # 유효 p value를 가지 Feature를 찾기 위해 번호를 붙이고 DataFrame 형태롤 변형
        b = np.array(b.sort_values([0], ascending=[True]))  # sort_value 함수로 오름차순
        # p-value 기반 상위 특징요소 선택하기
        New_Feature = np.zeros(shape=(2, self.Rank, np.size(FT, 2)))
        for i in np.arange(self.Rank):
            New_Feature[:, i, :] = np.array(FT[:, int(b[i, 1]), :])
            #진행상황3
            delay=delay+(i/Rank*10)
            pBar.setValue(delay)
            time.sleep(0.005)
        with open('./ref/New_Feature.pickle', 'wb') as f:
            pickle.dump(New_Feature, f, pickle.HIGHEST_PROTOCOL)
        with open('./ref/Select.pickle', 'wb') as f:
            pickle.dump(b, f, pickle.HIGHEST_PROTOCOL)

        # Target, Train 벡터 생성
        Total_target = np.zeros(shape=(np.size(New_Feature[:, 0, :])))
        Total_target[:np.size(New_Feature[0, 0, :])] = 1
        Train = np.concatenate((New_Feature[0, :, :], New_Feature[1, :, :]), axis=1)
        Train = np.transpose(Train)
        # print(np.size(Train[:,0]))
        # print(Train)

        # PNN 모델 생성 및 저장
        pnn = algorithms.PNN(std=10, verbose=False)
        pnn.train(Train, Total_target)
        with open('./ref/PNN_Model.pickle', 'wb') as f:
            pickle.dump(pnn, f, pickle.HIGHEST_PROTOCOL)
        # KNN 모델 생성 및 저장
        knn = KNeighborsClassifier()
        knn.fit(Train, Total_target)
        with open('./ref/KNN_Model.pickle', 'wb') as f:
            pickle.dump(knn, f, pickle.HIGHEST_PROTOCOL)
        # SVM 모델 생성 및 저장
        clf = svm.SVC()
        clf.fit(Train, Total_target)
        with open('./ref/SVM_Model.pickle', 'wb') as f:
            pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

        pBar.setValue(100)