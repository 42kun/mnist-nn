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

def softmax(Array):
    exps = np.exp(Array-np.max(Array,axis=1).reshape(Array.shape[0],1))
    return exps/np.sum(exps,axis=1).reshape(exps.shape[0],1)


def delta_cross_entropy(X,y):
    grad = softmax(X)-y
    return grad


def costJ(images,y,l2,thetaArray):
    """计算代价J"""
    m = y.shape[0]
    res = addOne(images)
    for i in range(len(thetaArray)):
        if i!=len(thetaArray)-1:
            res = addOne(sigmoid(res.dot(thetaArray[i])))
        else:
            res = softmax(sigmoid(res.dot(thetaArray[i])))
    #res = m*10
    J = np.sum(y*np.log(res)+(1-y)*np.log(1-res))/(-m)
    for i in thetaArray:
        J = J + l2*np.sum(i[:,1:]**2)/(2*m)
    return J

def normalize_image(image,mean=33.3,delta=255):
    return (image-mean)/delta


def nn_training(l,alpha_raw=0.001,l2=1e-4,epochs=10,batch_size=128):
    """输入学习率alpha,l2系数,
    每层神经元数量l，可以训练出整个神经网络并自动输出结果并保存参数
    请控制一下第一层为784（28*28），最后一层为10，我们训练集就是这样我也没办法"""

    data_train = Data(batch_size=batch_size,kind="train")
    data_test = Data(batch_size=batch_size,kind="t10k")
    layer = len(l)

    ###### 获取参数集合 ######
    thetaAll = 2 * np.random.rand(getThetaNumber(l)) - 1  # 获取参数集合，将来会将其分解为参数矩阵
    thetaArray = getThetaArray(l, thetaAll)  # 获取参数矩阵
    """如784，25，25，10，能分解为785*25 26*25 26*10 """

    tol = 0
    LOSS = []
    PRECISION = []
    for epo in range(epochs):
        for bat,(img,lab) in enumerate(data_train):
            img = normalize_image(img)
            y = getYArray(batch_size,lab)#m*10

            ####### 自适应学习率 #######
            # alpha = alpha_raw / (1 + 0.01*tol)
            alpha = alpha_raw
            ###### 向前传播 ######
            temp = addOne(img)  # i = 785*800 801*10
            a = [temp]
            for i in range(layer-1):
                if i != layer-2:
                    temp = addOne(sigmoid(temp.dot(thetaArray[i])))
                else:
                    temp = softmax(temp.dot(thetaArray[i]))
                a.append(temp)
            """
            (128, 785)
            (128, 801)
            (128, 11)
            """
            ###### 计算误差 ######
            # 交叉熵误差
            temp = a[-1]-y #难以想象，就是这么简单（可以严格推导）
            error = []
            # sigmoid误差
            for i in range(layer-1):
                error.append(temp)
                if i == layer-2:
                    break
                temp = subOne(temp.dot(thetaArray[layer-2-i].T)*(a[layer-2-i]*(1-a[layer-2-i])))

            ####### 计算梯度 #######
            grad = []
            for i in range(len(thetaArray)):
                grad.append(a[layer - 2 - i].T.dot(error[i]) / batch_size + l2*thetaArray[layer - 2 - i] / batch_size)

            ####### 更新参数 ######
            for i in range(layer - 1):
                thetaArray[i] = thetaArray[i] - alpha * grad[layer - 2 - i]

            if bat%100==0 and bat:
                tol += 1
                cost = costJ(img, y, l2, thetaArray)
                print("第%d次训练代价为:" % bat, cost)
                LOSS.append(cost)

        precision = []
        for img,lab in data_test:
            img = normalize_image(img)
            res = img
            for i in thetaArray:
                res = sigmoid(addOne(res).dot(i))
            res = softmax(res)
            res = res.argmax(1)
            pre = 100 * np.mean(res == lab)
            precision.append(pre)
        precision = np.array(precision).mean()
        PRECISION.append(precision)
        print("%d epoch 测试准确率为%.1f"%(epo,precision))


    thetaAll = packArray(thetaArray)
    np.save("theta.npy",thetaAll)
    print("theta已保存")
    import pickle
    with open("d.awa","wb") as fp:
        pickle.dump({"loss":LOSS,"pre":PRECISION},fp)

def test(l):#测试准确率函数
    print("\n开始计算准确率\n")
    timages,tlabels = load_mnist("mnist", "t10k")#获取测试集
    timages = normalize_image(timages)
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
    wrong = np.where(res != tlabels)[0]
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
    plt.savefig("wrong.png")
    plt.show()


if __name__ == "__main__":
    l = 784, 800, 10
    print("开始训练")
    # nn_training(l,l2=0.001,alpha_raw=0.1,epochs=20,batch_size=64)
    test(l)
    print("OK")

