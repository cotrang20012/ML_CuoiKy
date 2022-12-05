import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
import cv2
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_option('deprecation.showPyplotGlobalUse', False)

#GiamDanDaoHam
#bai01
def grad(x):
    return 2*x +5*np.cos(x)

def cost(x):
    return x**2 + 5*np.sin(x)

def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3: # just a small number
            break
        x.append(x_new)
    return (x, it)

(x1, it1) = myGD1(-5, .1)
GDDH01 = ('x= %.4f,cost=%.4f và số lần lập = %d' % (x1[-1],cost(x1[-1]),it1))
def mainGDDH01(k,fig):
    x =  np.linspace(-6,6,100)
    y = x**2 + 5*np.sin(x)
    plt.subplot(2,4,1)
    plt.plot(x,y,"b")
    plt.plot(x1[k],cost(x1[k]),'ro')
    s = 'iter %d/%d, grad=%.4f' % (k,it1,grad(x1[k]))
    plt.xlabel(s,fontsize=8)
    fig =  plt.tight_layout()
    st.pyplot(fig)

figGDDH01 = 0
st.title('Giảm dần đạo hàm ')
st.subheader('bai01')
kGDDH01 = st.slider('Chọn k', min_value=1, max_value=11, value=1, step=1)
number01 = int(kGDDH01)
st.caption(GDDH01)
mainGDDH01(number01,figGDDH01)
#bai02 
st.subheader('bai02')
def mainGDDH02():
    ax = plt.axes(projection="3d")
    x = np.linspace(-2,2,21)
    y = np.linspace(-2,2,21)
    X, Y = np.meshgrid(x,y)
    Z = X**2 + Y**2
    ax.plot_wireframe(X,Y,Z)
    fig = plt.show()
    st.pyplot(fig)
mainGDDH02()
#bai03
def mainGDDH03():
    X = np.random.rand(1000)
    y= 4+3*X +.5*np.random.randn(1000)
    plt.plot(X,y,'bo',markersize=2)
    #Mang mot chieu chuyen thanh ma tran
    y = np.array([y])
    X = np.array([X])
    #Chuyen vi ma tran
    X = X.T
    y = y.T
    model = LinearRegression()
    model.fit(X,y)
    w0 = model.intercept_
    w1 = model.coef_[0]
    x0 = 0
    y0 = w1*x0 + w0
    x1 = 1
    y1 = w1*x1 + w0
    plt.plot([x0,x1],[y0,y1],'r')
    fig = plt.show()
    st.pyplot(fig)
st.subheader('bai03 ')
mainGDDH03()
######################################################

#HoiQuyDaThuc
N = 30
N_test = 30
#bai01
def mainHQDT01():
    np.random.seed(100)
    X = np.random.rand(N,1)*5
    y = 3*(X-2) * (X-3)*(X-4) + 10*np.random.randn(N,1)

    X_true = np.linspace(0,5,51)
    y_true = 3*(X_true-2) * (X_true-3)*(X_true-4)

    X_test = (np.random.randn(N_test,1) - 1/8)*10
    y_test = 3*(X_test -2)*(X_test-3)*(X_test-4)+10*np.random.randn(N,1)


    plt.plot(X,y,'ro',markersize=3)
    plt.plot(X_true,y_true,'y') 
    
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    # for i in range(N):
    #     print('%10.4f, %10.4f %10.4f'%(X[i,0],X_poly[i,0],X_poly[i,1]))
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)
    w0 = lin_reg.intercept_[0]
    w1 = lin_reg.coef_[0,0]
    w2 = lin_reg.coef_[0,1]

    y_predict = w0 + w1*X_true + w2*X_true**2

    plt.plot(X_true,y_predict,'r')
    #tính sai số trên tập test
    X_test_poly = poly_features.fit_transform(X_test)
    y_test_predict = lin_reg.predict(X_test_poly)
    mse_test = mean_squared_error(y_test,y_test_predict)
    rmse_test = np.sqrt(mse_test)
    print('Sai số bình phương trung bình - test:')
    print('%.4f' % rmse_test)
    fig = plt.show()
    st.pyplot(fig)
    textHQDT01 = 'Sai số bình phương trung bình test: ' + '%.4f' % rmse_test
    st.subheader(textHQDT01)

st.title('Hồi quy đa thức ')
st.subheader('bai01')
mainHQDT01()
#bai02
def mainHQDT02(): 
    np.random.seed(100)
    X = np.random.rand(N,1)*5
    y = 3*(X-2) * (X-3)*(X-4) + 10*np.random.randn(N,1)

    X_true = np.linspace(0,5,51)
    y_true = 3*(X_true-2) * (X_true-3)*(X_true-4)

    X_test = (np.random.randn(N_test,1) - 1/8)*10
    y_test = 3*(X_test -2)*(X_test-3)*(X_test-4)+10*np.random.randn(N,1)


    plt.plot(X,y,'ro',markersize=3)
    plt.plot(X_true,y_true,'y')
    
    poly_features = PolynomialFeatures(degree=4, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    # for i in range(N):
    #     print('%10.4f, %10.4f %10.4f'%(X[i,0],X_poly[i,0],X_poly[i,1]))
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)
    w0 = lin_reg.intercept_[0]
    w1 = lin_reg.coef_[0,0]
    w2 = lin_reg.coef_[0,1]
    w3 = lin_reg.coef_[0,2]
    w4 = lin_reg.coef_[0,3]


    y_predict = w0 + w1*X_true + w2*X_true**2 +w3*X_true**3 + w4*X_true**4

    plt.plot(X_true,y_predict,'r')
    #tính sai số trên tập test
    X_test_poly = poly_features.fit_transform(X_test)
    y_test_predict = lin_reg.predict(X_test_poly)
    mse_test = mean_squared_error(y_test,y_test_predict)
    rmse_test = np.sqrt(mse_test)
    print('Sai số bình phương trung bình - test:')
    print('%.4f' % rmse_test)
    fig = plt.show()
    st.pyplot(fig)
    textHQDT02 = 'Sai số bình phương trung bình test: ' + '%.4f' % rmse_test
    st.subheader(textHQDT02)

st.subheader('bai02')
mainHQDT02()

#KNN
#knn = joblib.load("D:\HT\hk7\ML\KNN\knn_digit.pkl")
knn = joblib.load("knn_digit.pkl")
st.title('KNN')
st.subheader('bai02')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    image = Image.open(uploaded_file)
    image = np.array(image)
    #image = Image.frombuffer("L",(28,28),bytes_data,'raw',"L", 0, 1)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image,(28,28))
    image = image.reshape(1,784)
    st.image(uploaded_file)
    #print(bytes_data)
    predicted = knn.predict(image)
    text = 'Kết quả: ' + str(predicted[0])
    st.subheader(text)
######################################################
