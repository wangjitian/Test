# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:05:58 2019

@author: Design
"""

#网络模型的设计部分，分类网络在此设置。
def SharePart(input, drop_out_rate):
    
    def pre_process(input):
        rgb_scaled = input
        Mean = [103.939,116.779,123.68]
        
        red,green,blue = tf.split(rgb_scaled,3,3)
        bgr = tf.concat([
                red - Mean[2],
                green - Mean[1],
                blue - Mean[0]],3)
        return bgr
    
    input = pre_process(input)
    
    with tf.variable_scope('Share_Part'):
        
        conv1 = l.conv2d('conv1',input,(11,11),96,strides = [1,4,4,1],decay = (0.0,0.0),pad='VALID',Init = MODEL_INIT['conv1'])
        maxpool1 = l.max_pooling('maxpool',conv1,3,2)
        norm1 = tf.nn.lrn(maxpool1,depth_radius=2,alpha=2e-05,beta=0.75,name='conv1')
    
        conv2 = l.conv2d_with_group('conv2',norm1,(5,5),256,2,decay = (0.0,0.0),pad = 'SAME', Init = MODEL_INIT['conv2'])
        maxpool2 = l.max_pooling('maxpool2',conv2,3,2)
        norm2 = tf.nn.lrn(maxpool2,depth_radius=2,alpha=2e-05,beta=0.75,name='conv2')

        conv3 = l.conv2d('conv3',norm2,(3,3),384,pad = 'SAME',Init = MODEL_INIT['conv3'])
    
    
        conv4 = l.conv2d_with_group('conv4',conv3,(3,3),384,2,pad = 'SAME',Init = MODEL_INIT['conv4'])
       
        conv5 = l.conv2d_with_group('conv5',conv4,(3,3),256,2,pad = 'SAME',Init = MODEL_INIT['conv5'])
        maxpool5 = l.max_pooling('maxpool5',conv5,3,2)
        print (maxpool5.shape)
    
        dim=1
        shape = maxpool5.get_shape().as_list()
        for d in shape[1:]:
            dim*=d
    
        reshape = tf.reshape(maxpool5,[-1,dim])
    
        fc6 = l.fully_connect('fc6',reshape,4096,Init = MODEL_INIT['fc6'])
        fc6 = l.dropout('drop_6',fc6,drop_out_rate)
        fc7 = l.fully_connect('fc7',fc6,4096,Init = MODEL_INIT['fc7'])
        fc7 = l.dropout('drop_7',fc7,drop_out_rate)
        
    return fc7

#网络的任务层也就是softmax层
def MissionPart(input):
    
    with tf.variable_scope('Classifier'):
        result = l.fully_connect('classifier',input,CLASS_NUM,active=None)
    return result

#网络层计算损失函数部分
def SoftmaxWithLoss(logistic,label):
    
    label = tf.one_hot(label,depth = CLASS_NUM)
    loss = tf.losses.softmax_cross_entropy(label,logistic)
    
    return loss

#训练网络时所调节的网络层参数以及优化方法
def MMDPart(input1,input2,mmd_alpha = 0.3):
    with tf.variable_scope('MMD'):
        for i in simple_num:
            size1 = input1.size()
            size2 = input2.size()
            s1 = random.randint(0,size1)
            s2 = random.randint(0,size1)
            s2 = s2+1 if s1 == s2 else s2
            t1 = random.randint(0,size2)
            t2 = random.randint(0,size2)
            t2 = t2+1 if t2 == t1 else t2
            x1 = input1[s1], x2 = input1[s2], y1 = input2[t1], y2 = input2[t2]
            tmp1 = tf.subtract(x1,x2)
            tmp1 = tf.tensordot(tmp1,tmp1)
            tmp2 = tf.subtract(y1,y2)
            tmp2 = tf.tensordot(tmp2,tmp2)
            tmp3 = tf.subtract(x1,y2)
            tmp3 = tf.tensordot(tmp3,tmp3)
            tmp4 = tf.subtract(x2,y1)
            tmp4 = tf.tensordot(tmp4,tmp4)
            mmd = tmp1+tmp2-tmp3-tmp4
def train_net(loss,base_lr=0.00001):
    
    
    var_list = tf.trainable_variables()
    trn_list = []
    for i in var_list:
        if 'conv1' not in i.name and 'conv2' not in i.name:
            trn_list.append(i)
            tf.summary.histogram('weight',i)
    
    loss = tf.add_n(tf.get_collection('losses'),name='all_loss')
    opt = tf.train.AdamOptimizer(base_lr).minimize(loss,var_list=trn_list)
    return opt

#测试函数
def Test(logistic,label):
    
    result = tf.cast(tf.argmax(logistic,axis = 1),tf.uint8)
    compare = tf.cast(tf.equal(result,label),tf.float32)
    acc = tf.reduce_mean(compare)
    return acc