# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:03:56 2019

@author: Design
"""


from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import matplotlib as mat
import numpy as np
#import tensorflow as tf

zhfont1 = mat.font_manager.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')

def plot_confusion_matrix(cm, title):
    cm_o = cm
    plt.rcParams['figure.figsize'] = (12.0, 10.0)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title,fontproperties =zhfont1, fontsize=40)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(20))    
    plt.xticks(num_local, num_local, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, num_local)    # 将标签印在y轴坐标上
    plt.ylabel('真实标签',fontproperties =zhfont1, fontsize=40)    
    plt.xlabel('预测标签',fontproperties =zhfont1, fontsize=40)
    for i in num_local:
#        for j in num_local:
        plt.text(i, i, str('%.2f' % (cm[i, i])), va='center', ha='center')

cm1 = np.load('D:/王继天学习资料/ITIS2/tvt实验/confusion_matrix/comps_sv_20(3)_3_pre.npy')
plot_confusion_matrix(cm1, "初始阶段")
plt.savefig('D:\王继天学习资料\code\python/毕设_BT0.png', format='png')
plt.show()

cm1 = np.load('D:/王继天学习资料/ITIS2/tvt实验/confusion_matrix/comps_sv_20(3)_3_ALBT_1.npy')
plot_confusion_matrix(cm1, "第一次学习阶段")
plt.savefig('D:\王继天学习资料\code\python/毕设_BT1.png', format='png')
plt.show()

cm1 = np.load('D:/王继天学习资料/ITIS2/tvt实验/confusion_matrix/comps_sv_20(3)_3_ALBT_2.npy')
plot_confusion_matrix(cm1, "第二次学习阶段")
plt.savefig('D:\王继天学习资料\code\python/毕设_BT2.png', format='png')
plt.show()


cm1 = np.load('D:/王继天学习资料/ITIS2/tvt实验/confusion_matrix/comps_sv_20(3)_3_ALBT_3.npy')
plot_confusion_matrix(cm1, "第三次学习阶段")
plt.savefig('D:\王继天学习资料\code\python/毕设_BT3.png', format='png')
plt.show()

cm1 = np.load('D:/王继天学习资料/ITIS2/tvt实验/confusion_matrix/comps_sv_20(3)_3_ALBT_4.npy')
plot_confusion_matrix(cm1, "第四次学习阶段")
plt.savefig('D:\王继天学习资料\code\python/毕设_BT4.png', format='png')
plt.show()

cm1 = np.load('D:/王继天学习资料/ITIS2/tvt实验/confusion_matrix/comps_sv_20(3)_3_ALBT_5.npy')
plot_confusion_matrix(cm1, "第五次学习阶段")
plt.savefig('D:\王继天学习资料\code\python/毕设_BT5.png', format='png')
plt.show()

confunsion_matix = []
file_report = open('pre_'+'report.txt','w')
recall = []
precision = []
f1 = []
support = []
accuracy = 0
for i in range(20):
    accuracy += cm1[i,i]
    try:
        recall.append(round(cm1[i,i]/np.sum(cm1[i]),3))
    except:
        recall.apprend(round(0,3))
    try:
        precision.append(round(cm1[i,i]/np.sum(cm1[:,i]),3))
    except:
        precision.append(round(0,3))
    try:
        f1.append(round(2*recall[i]*precision[i]/(recall[i]+precision[i]),3))
    except:
        f1.append(round(0,3))
    support.append(np.sum(cm1[i]))
    file_report.write(str(i).rjust(10,)+str(precision[i]).rjust(10,)+str(recall[i]).rjust(10,)+str(f1[i]).rjust(10,)+str(support[i]).rjust(10,)+'\n')
try:
    recall_avg = round(np.sum(np.array(recall))/20,3)
except:
    recall_avg = 0
try:
    precision_avg = round(np.sum(np.array(precision))/20,3)
except:
    precision_avg = 0
try:
    f1_avg = round(np.sum(np.array(f1))/20,3)
except:
    f1_avg = 0
support_num = np.sum(np.array(support))
accuracy = round(accuracy/support_num,5)
file_report.write("average".rjust(10,)+str(precision_avg).rjust(10,)+str(recall_avg).rjust(10,)+str(f1_avg).rjust(10,)+str(support_num).rjust(10,)+'\n')
file_report.write(" stage acc is " +str(accuracy))
file_report.write("\n\n\n\n")
file_report.close()