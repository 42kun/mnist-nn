import numpy as np
import struct
import random
import os

def load_mnist(path,kind = "train"):
    image_path = os.path.join(path,"%s-images-idx3-ubyte"%kind)
    lable_path = os.path.join(path,"%s-labels-idx1-ubyte"%kind)
    with open(image_path,"rb") as imgp:
        magic,num,row,col = struct.unpack(">IIII",imgp.read(16))#文件内部指针已偏移
        #其实struct不用也没关系，只要让文件指针偏移就够了
        images = np.fromfile(imgp,dtype=np.uint8).reshape(num,row*col)#images = 60000*784
        #np.savetxt("images.txt",images,fmt=  '%.1e')#将数据保存在文本文件中，好像并没什么卵用
        #print(np.size(images, 0))
        #print(np.size(images, 1))#输出行数/列数
    with open(lable_path,"rb") as labp:
        labp.read(8)
        labels = np.fromfile(labp,dtype=np.uint8)#lables = 5000*1
    return images,labels

class Data(object):
    def __init__(self,kind="train",batch_size=128,shuffle=True):
        self.images,self.labels = load_mnist("mnist","train")
        self.size = len(self.images)
        self.batch_size = batch_size
        self.p = [i for i in range(self.size)]
        self.n = 0
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.p)


    def __iter__(self):
        return self

    def __next__(self):
        if self.n+self.batch_size<self.size:
            imgt = self.images[self.p[self.n:self.n+self.batch_size]]
            labt = self.labels[self.p[self.n:self.n+self.batch_size]]
            self.n+=self.batch_size
            return imgt,labt
        else:
            self.n=0
            if self.shuffle:
                random.shuffle(self.p)
            raise StopIteration()


if __name__ == "__main__":
    for a,b in Data():
        print(b)
    print("@")
    for a,b in Data():
        print(a.shape)