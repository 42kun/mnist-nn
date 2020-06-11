#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：BasicEnvironment -> myPicture
@IDE    ：PyCharm
@Author ：yuexiahu
@Date   ：2020/5/20 23:07
@Desc   ：
=================================================='''
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.ticker import  MultipleLocator

dpi = 1000
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS','SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = dpi #图片像素
plt.rcParams['figure.dpi'] = dpi #分辨率
plt.rcParams["font.family"] = 'Times New Roman'

label_y = "Precision"

with open("../mnist/d.awa","rb") as fp:
    dm = pickle.load(fp)
pre = dm["pre"]
loss = dm["loss"]

# dim
# data_x_name = [2,4,8,16,32,64]
data_x = [i for i in range(len(pre))]
data_y = pre
label_x = "epochs"
m = 'D'
c = "#4BBDE4"


marker2 = ['.','v','D','*','s','d','p','o']
color = ['#CC00CC','#66CC00','#CC3300',"#0000FF","#FF9900","#4BBDE4","#871F78","#FF4500"]

LABEL_SIZE = 8
TICKET_SIZE = 6
LEGEND_SIZE = 5
GRID_SPACE = 4
GRID_WIDTH = 1
bwith = 1.1
tick = 5

plt.figure(figsize=(4,3))
# 设置图标基础样式
# plt.xlim(10,50)
plt.xticks(data_x,size = TICKET_SIZE)

# # plt.ylim(0.2,0.82)
plt.yticks(size = TICKET_SIZE)
# ra = (0.2,0.8)
# plt.ylim(ra[0],ra[1])
# plt.yticks(np.arange(ra[0],ra[1],0.05),size = TICKET_SIZE)
# plt.grid(True)
plt.grid(color='gray',linestyle='-',axis='y',dashes=(GRID_SPACE,GRID_SPACE),lw=GRID_WIDTH)


ax = plt.gca()#获取边框
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
ax.tick_params(axis="x",direction="in")
ax.tick_params(axis="y",length=0)
# xmajorLocator = MultipleLocator(20)
# ax.xaxis.set_major_locator(xmajorLocator)
# ax.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度

plt.plot(data_x,data_y,marker = m,linewidth=1,fillstyle="none",markersize=4,c=c)
# print(marker2[i])


# plt.legend(exp_name3,loc=3,fontsize=LEGEND_SIZE)
plt.ylabel(label_y,fontsize=LABEL_SIZE)
plt.xlabel(label_x,fontsize=LABEL_SIZE)
plt.savefig("%s.png"%(label_x))
plt.show()