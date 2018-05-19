#coding=UTF-8
import numpy as np
from numpy import *
def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return  dataMat,labelMat

def standRegress(xArr,yArr):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)
    xTx=xMat.T*xMat

    if np.linalg.det(xTx)==0.0:
        print "矩阵没有逆矩阵"

    ws=xTx.I*(xMat.T*yMat)
    return  ws

#局部加权线性回归函数
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):  # next 2 lines create weights matrix
        diffMat = testPoint - xMat[j, :]  #
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)

    if np.linalg.det(xTx)==0.0:
        print "矩阵没有逆矩阵"

    ws=xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(mat(testArr))[0]
    yHat=zeros(m)
    for i in  range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat

#岭回归
def ridgeRegress(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if np.linalg.det(denom)==0.0:
        print "矩阵没有逆矩阵"
    ws=denom.I*(xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    #数据标准化
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    xVar=var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegress(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def regularize(xMat):
    inMat=xMat.copy()
    inMeans = mean(inMat, 0)  # calc mean then subtract it off
    inVar = var(inMat, 0)  # calc variance of Xi then divide by it
    inMat = (inMat - inMeans) / inVar
    return inMat
#前向逐步线性回归
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yArr,0)
    yMat=yMat-yMean
    xMat=regularize(xMat)
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))
    ws=zeros((n,1))
    wsTest=ws.copy()
    wsMax=ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError=inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat
#测试前向逐步线性回归
# xArr,yArr=loadDataSet('abalone.txt')
# stageWise=stageWise(xArr,yArr,0.001,5000)
# import matplotlib.pyplot as plt
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.plot(stageWise)
# plt.show()


#测试
# xArr,yArr=loadDataSet("ex0.txt")
#对单点进行测试

#print lwlr(mat(xArr[0]),xArr,yArr,1.0)
#数据的所有点进行测试,k=1 所有的数据等权重，k=0.01  可以挖掘出数据的潜在规律,k=0.003考虑
#了过多的噪声，导致数据过拟合

# #预测的结果
# yHat=lwlrTest(xArr,xArr,yArr,0.003)
# #用图像看YHat的拟合效果
# #所用的图像函数需要将数据点安序排列
# xMat=mat(xArr)
# sortIndex=xMat[:,1].argsort(0)
# xSort=xMat[sortIndex][:,0,:]
# import matplotlib.pyplot as plt
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.plot(xSort[:,1],yHat[sortIndex])
# ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
# plt.show()

#测试：岭回归对鲍鱼年龄进行预测
# abX,abY=loadDataSet('abalone.txt')
# ridgeWeights=ridgeTest(abX,abY)
# import matplotlib.pyplot as plt
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.plot(ridgeWeights[0])
# plt.show()
