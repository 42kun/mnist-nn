from mnist.getData import *
import numpy as np
import matplotlib.pyplot as plt
import random
from mnist.Discern import *

def addOne(x):#将数组前加一列1
    return np.concatenate((np.ones((x.shape[0],1)),x),1)

def subOne(x):#将数组减去第一列
    return x[:,1:]

def getThetaNumber(l):#获取所有需要的参数个数
    '''400*25*10 theta = 25*401 + 10 *26'''
    sum = 0
    for i in range(len(l)-1):
        sum = sum + (l[i]+1)*l[i+1]
    return sum

def getYArray(m,lables):#输入标签个数与标签值矩阵，生成结果矩阵y
    """生成m*10矩阵，每行仅结果处为1，其余为0"""
    y = np.zeros((m,10))
    for i in range(m):
        y[i,lables[i]]=1
    return y

def getThetaArray(l,thetaAll):
    """将一个参数向量拆分成若干个参数数组
       比如说l = 400,25,10
       它能将thetaAll分解为401*25
       26*10"""
    thetaArray = []
    star = 0
    for i in range(len(l)-1):
        thetaArray.append(thetaAll[star:star+((l[i]+1)*l[i+1])].reshape((l[i]+1,l[i+1])))
        star = star + (l[i]+1)*l[i+1]
    return thetaArray

def packArray(thetaArray):
    """将若干个数组打包成参数向量"""
    thetaAll = np.array([])
    for i in thetaArray:
        thetaAll = np.concatenate((thetaAll,i.reshape(i.size)))
    return thetaAll

def sigmoid(Array):
    return 1.0/(1.0+np.exp(-Array))

def costJ(images,y,lamb,thetaArray):
    """计算代价J"""
    m = y.shape[0]
    res = images
    for i in thetaArray:
        res = sigmoid(addOne(res).dot(i))
    #res = m*10
    J = np.sum(y*np.log(res)+(1-y)*np.log(1-res))/(-m)
    for i in thetaArray:
        J = J + lamb*np.sum(i[:,1:]**2)/(2*m)
    return J


def nn_training(alpha_raw=0.1,lamb=1,iter=50,layer=4,l=None):
    """输入学习率alpha,偏置lamb,层数layer，迭代次数iter
    每层神经元数量l，可以训练出整个神经网络并自动输出结果并保存参数
    请控制一下第一层为784（28*28），最后一层为10，我们训练集就是这样我也没办法"""
    if(layer!=len(l)):
        print("层数与参数个数不匹配")
        print("退出")
        return

    images,labels = load_minist("mnist", "train")#获取训练集
    images = (images - 33.3) / 255#正规化
    m = images.shape[0]#训练集个数
    n = images.shape[1]#图像像素数
    thetaAll = 2*np.random.rand(getThetaNumber(l))-1#获取参数集合，将来会将其分解为参数矩阵
    y = getYArray(m,labels)#m*10

    ####### 计算代价 #######
    thetaArray = getThetaArray(l,thetaAll)#获取参数矩阵
    """如784，25，25，10，能分解为785*25 26*25 26*10 """
    #print("原始代价为:",costJ(images,y,lamb,thetaArray))

    ####### 开始迭代 #######
    for _ in range(iter):
        ####### 自适应学习率 #######
        alpha = alpha_raw/(1+0.001*iter)
        ####### 计算全部的a #######
        ### num(a) = layer, 比如784，25，25，10时a有4个,a1=images,a2,a3,a4 ###
        temp = addOne(images) #m*785 i = 785*25 26*25 26*10
        a = [temp]
        for i in thetaArray:
            temp = addOne(sigmoid(temp.dot(i)))
            a.append(temp)
        a[len(a)-1] = subOne(a[len(a)-1]) #a = m*785 m*26 m*26 m*10

        ####### 计算全部的误差 #######
        ### error是反着的，分别为e4，e3，e2 m*10 m*25 m*25,e1不计算误差 ###
        temp = a[len(a)-1]-y
        error = []
        for i in range(layer-1):
            error.append(temp)
            if i == layer-2:
                break
            temp = subOne(temp.dot(thetaArray[layer-2-i].T)*(a[layer-2-i]*(1-a[layer-2-i])))
        ####### error计算完成 #######

        ####### 计算梯度 #######
        grad = []
        for i in range(len(thetaArray)):
            grad.append(a[layer - 2 - i].T.dot(error[i])/m+lamb*thetaArray[layer-2-i]/m)
        ####### 梯度计算完成,分别算出t3，t2，t1#######

        for i in range(layer-1):
            thetaArray[i] = thetaArray[i] - alpha*grad[layer-2-i]

        if _%100==0:
            print("第%d次训练代价为:"%_, costJ(images, y, lamb, thetaArray))
    SGD
    mnist
    numpy

    thetaAll = packArray(thetaArray)
    np.save("theta.npy",thetaAll)
    print("theta已保存")

def test(l):#测试准确率函数
    print("\n开始计算准确率\n")
    timages,tlabels = load_minist("mnist", "t10k")#获取测试集
    timages = (timages - 33.31) / 255#正规化
    print("导入测试集成功\n")
    thetaAll = np.load("theta.npy")
    thetaArray = getThetaArray(l,thetaAll)

    res = timages
    for i in thetaArray:
        res = sigmoid(addOne(res).dot(i))
    res = res.argmax(1)
    precision = 100*np.mean(res == tlabels)
    print("准确率为:",precision)
    ###### 以下是随机生成几张图片 ######
    pictures = timages[random.randint(0, 10000)]

    img = pictures.reshape(28, 28)
    plt.imshow(img,cmap='Greys')
    plt.axis('off')
    plt.savefig("test.jpg")

    ###### 以下是绘制出错误样本 ######
    wrong = np.where(res == tlabels)[0]
    wrong_index = wrong[[random.randint(0, wrong.size) for _ in range(10)]]
    wrong = timages[wrong_index]
    wrong_label = res[wrong_index]
    fig, ax = plt.subplots(2, 5)
    ax = ax.flatten()
    for i in range(10):
        img = wrong[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')
        ax[i].set_title(str(wrong_label[i]))
    fig.suptitle("recognize result")
    plt.show()


if __name__ == "__main__":
    l = 784, 800, 10
    if os.path.exists("theta.npy"):
        print("参数已存在，无需训练")
        dis = Discern()
        thetaAll = np.load("theta.npy")
        thetaArray = getThetaArray(l, thetaAll)
        res = np.array(dis.img).reshape(1,784)

        res = 255 - res
        res = (res-33.31)/255
        for i in thetaArray:
            res = sigmoid(addOne(res).dot(i))
        res = res.argmax(1)
        print("这个数字是")
        print(res)
    else:
        print("开始训练")
        nn_training(2,4,1200,len(l),l)
        test(l)
        print("OK")

