# SDML
Sungkunkwan_University_SDML

Test_for_github from SKKU_SDML

성균관대학교 지속가능 설계 및 생산 연구실 PHM CODE

Modules
 ```
#needs for matrix calculation & statistic calculate
pip install numpy
pip install scipy
pip install pandas

#needs for wavelet transform
pip install pywt

#needs for machine-learning (knn,pnn,svm)
pip install scikit-learn
pip install neupy

#needs for plot graph
pip install matplotlib
pip install seaborn
```

#1. PHM File

#데이터 특징요소 추출

Wavelet 영역의 Feature 들을 추출 + Time-Domain 에서의 Feature 추출

```
PHM
├ Data 특징요소 추출
├ P-value 추출
├ K-Fold 및 검증
├ 학습모델 생성 (pickle 파일로 생성)
├ Class_Transform (진단 할 데이터를 학습한 Feature에 맞게 변형)
└ └ 
```

#2.GUI

#데이터를 추출하고 고장진단을 하는 GUI

