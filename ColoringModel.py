
# coding: utf-8

# In[1]:


'''
import torch
import torchvision
'''
'''
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
'''
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
from glob import glob
import scipy as sp
import sys
from PIL import Image


# In[2]:


#import cPickle as pickle
import tensorflow as tf
import os
from PIL import Image, ImageCms
import glob
import cv2


# In[ ]:


##################code only for generating a new gray file###########no need to run now

color_image_path='./test_image/'
gray_image_path='./gray_image/'
def add_gray_file(color_image_path,gray_image_path):
    index=0
    for filename in os.listdir(color_image_path):
        img = Image.open(color_image_path+filename)
        g_img=img.convert('LA')
        g_img.save(gray_image_path+filename[:-4]+'.png')
        index+=1
        
add_gray_file(color_image_path,gray_image_path)


# In[4]:


def get_image(color_image_path,gray_image_path):
    image=[]
    gray_image=[]
    original_img=[]
    for filename in os.listdir(color_image_path):
        img = Image.open(color_image_path+filename)
        #print(filename_g)
        g_img = Image.open(gray_image_path+filename[:-4]+'.png')
        image.append(np.array(img).astype(float))
        gray_image.append(np.array(g_img).astype(float)[:,:,0])
        original_img.append(np.array(img))
    return np.array(image),np.array(gray_image),np.array(original_img)

color_image_path='./test_image/'
gray_image_path='./gray_image/'
image,gray_image,o_img = get_image(color_image_path,gray_image_path)


# In[8]:


upper = 100.
lower = -upper+1
n = 20.
unit = (upper-lower+1)/n

def lab_2_color(lab):
    lab[:,:,1] = np.maximum(lab[:,:,1], lower)
    lab[:,:,1] = np.minimum(lab[:,:,1], upper)
    
    lab[:,:,2] = np.maximum(lab[:,:,2], lower)
    lab[:,:,2] = np.minimum(lab[:,:,2], upper)
    
    x, y = lab[:,:,1], lab[:,:,2]
    i, j = (x.astype(int) - lower) / unit , (y.astype(int) - lower) / unit
    color = i.astype(int) * n + j.astype(int)
    color.astype(int)
    return color

def color_2_lab(color, L): 
    color = np.maximum(color, 0)
    color = np.minimum(color, n*n-1)
    a = color / n * unit + lower
    b = color % n * unit + lower
    
    lab = np.zeros((a.shape[0],a.shape[1],3))
    lab[:,:,0] = L
    lab[:,:,1] = a
    lab[:,:,2] = b
    return lab


# In[11]:


def get_lab_for_all(path,num):  
    lab_all=[]
    count=0
    for filename in os.listdir(path):
        if num!='all' and count>num:
            break
        #print(count)
        count+=1
        
        image_path=path+filename
        im = Image.open(image_path)
        
        if im.mode != "RGB":
            im = im.convert("RGB").resize((224,224))
        
        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile  = ImageCms.createProfile("LAB")
        
        rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
        lab_im = ImageCms.applyTransform(im, rgb2lab_transform)
        lab_all.append(np.array(lab_im))
    return np.array(lab_all)


# In[12]:


path = './test_image/'
lab_all=get_lab_for_all(path,'all')


# In[6]:


###########seperate the data
#gray_image=np.array(gray_image)[:,:,:,0]
r_image=np.array(image)[:,:,:,0]
g_image=np.array(image)[:,:,:,1]
b_image=np.array(image)[:,:,:,2]


# In[7]:


############with shape [batchsize,height,width,channel]
def generate_image(r,g,b):
    r=tf.reshape(r,[-1,224,224,1])
    g=tf.reshape(g,[-1,224,224,1])
    b=tf.reshape(b,[-1,224,224,1])   


    total_image=tf.concat([r,g],3)
    total_image=tf.concat([total_image,b],3)
    '''
    plt.imshow(np.array(image).astype('uint8'))
    plt.show()
    '''
    return total_image


# In[8]:


def np_generate_image(r,g,b):
    r=np.reshape(r,[-1,224,224,1])
    g=np.reshape(g,[-1,224,224,1])
    b=np.reshape(b,[-1,224,224,1])
    
    total_image=np.concatenate([r,g],3)
    total_image=np.concatenate([total_image,b],3)
    '''
    plt.imshow(np.array(image).astype('uint8'))
    plt.show()
    '''
    return total_image


# In[13]:


################get a set of ab color as label, and illuminat as gray image
lab2color_label=[]
l_img=[]
for l in lab_all:
    lab2color_label.append(lab_2_color(l))
    l_img.append(l[:,:,0])
lab2color_label=np.array(lab2color_label)
l_img=np.array(l_img)

################get 2-dimension ab map
lab_label_2=lab_all[:,:,:,-2:]
lab_label_2=np.array(lab_label_2)


# In[19]:


plt.imshow(lab_label_2[1][:,:,0])
plt.show()


# In[20]:


plt.imshow(lab_label_2[1][:,:,1])
plt.show()


# In[53]:


plt.imshow(l_img[1],cmap='gray')
plt.show()


# In[3]:


'''''''''''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
run the code starting from here
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''''''''''
def cv2_get_lab_for_all(path,num):  
    lab_all=[]
    count=0
    for filename in os.listdir(path):
        if num!='all' and count>num:
            break
        #print(count)
        count+=1
        
        image_path=path+filename
        im = Image.open(image_path)
        ############################
        '''
        im.thumbnail([400,400], Image.ANTIALIAS)
        w,h = im.size
        w/=4
        h/=4
        im = im.crop((w,h,w+224,h+224))
        '''
        im = im.resize([224,224], Image.ANTIALIAS)
        ##############strange error occur when reading these image
        if count in [607,875,1189,2765,3833,4114,4530,4907]:
            continue
        lab = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2LAB).astype("float32")
        #(l, a, b) = cv2.split(lab)
        print(count)
        # clip the pixel intensities to [0, 255] if they fall outside
        # this range
        #l = np.clip(l, 0, 255)
        #a = np.clip(a, 0, 255)
        #b = np.clip(b, 0, 255)

        #transfer = cv2.merge([l, a, b])
        lab_all.append(np.array(lab))
    return np.array(lab_all)


# In[4]:


def cv2_get_lab_for_all_human(path,num):  
    lab_all=[]
    count=0
    for filepath in os.listdir(path):
        if num!='all' and count>num:
            break
        #print(count)
        
        for filename in os.listdir(path+filepath):
            count+=1
            image_path=path+filepath+'/'+filename
            im = Image.open(image_path)
            ############################
            '''
            im.thumbnail([400,400], Image.ANTIALIAS)
            w,h = im.size
            w/=4
            h/=4
            im = im.crop((w,h,w+224,h+224))
            '''
            im = im.resize([224,224], Image.ANTIALIAS)
            ##############strange error occur when reading these image
            if count in [607,875,1189,2765,3833,4114,4530,4907]:
                continue
            lab = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2LAB).astype("float32")
            #(l, a, b) = cv2.split(lab)
            print(count)
            # clip the pixel intensities to [0, 255] if they fall outside
            # this range
            #l = np.clip(l, 0, 255)
            #a = np.clip(a, 0, 255)
            #b = np.clip(b, 0, 255)

            #transfer = cv2.merge([l, a, b])
            lab_all.append(np.array(lab))
    return np.array(lab_all)


# In[35]:


###################get LAB from cv2
path = './train_image/train/'
lab_all=cv2_get_lab_for_all(path,200)


# In[36]:


def get_i_o(lab_all):
    lab_label_2=lab_all[:,:,:,-2:]
    lab_label_2=np.array(lab_label_2)
    l_img = lab_all[:,:,:,0]
    return l_img,lab_label_2


# In[37]:


def redefine_loss(logits, color):
    '''
    logits_flat = tf.reshape(logits, [-1, 224*224])
    color_flat = tf.reshape(color, [-1, 224*224])
    
    predict=logits_flat
    target=color_flat
    '''
    return tf.losses.mean_squared_error(color,logits)
    #return tf.losses.mean_squared_error(target,predict)


# In[51]:


def encoder(gray_image1,gray_image2,dec,act_mode,state=True):
    ############gray_image1=gray_image2, we only use 2 when train=false
    #gray_image=tf.reshape(gray_image,[-1,224,224,1])
    print (gray_image1.shape)
    #################block 1
    layer_1 = tf.layers.conv2d(gray_image1,filters=64,kernel_size=3,strides=1,padding='SAME',activation=act_mode)
    layer_1 = tf.contrib.layers.batch_norm(layer_1,decay=dec,is_training=state)
    
    layer_2 = tf.layers.conv2d(layer_1,filters=64,kernel_size=3,strides=2,padding='SAME',activation=act_mode)
    layer_2 = tf.contrib.layers.batch_norm(layer_2,decay=dec,is_training=state)
    #print(layer_1.shape)
    
    #################block 2
    # Hidden fully connected layer with 128 neurons
    layer_3 = tf.layers.conv2d(layer_2,filters=128,kernel_size=3,strides=1,padding='SAME',activation=act_mode)
    layer_3 = tf.contrib.layers.batch_norm(layer_3,decay=dec,is_training=state)
    
    layer_4 = tf.layers.conv2d(layer_3,filters=128,kernel_size=3,strides=2,padding='SAME',activation=act_mode)
    layer_5 = tf.contrib.layers.batch_norm(layer_4,decay=dec,is_training=state)
    #print(layer_3.shape)
    
    #################block 3
    # Hidden fully connected layer with 256 neurons
    layer_5 = tf.layers.conv2d(layer_4,filters=256,kernel_size=3,strides=1,padding='SAME',activation=act_mode)
    layer_5 = tf.contrib.layers.batch_norm(layer_5,decay=dec,is_training=state)
    
    layer_6 = tf.layers.conv2d(layer_5,filters=512,kernel_size=3,strides=2,padding='SAME',activation=act_mode)
    layer_6 = tf.contrib.layers.batch_norm(layer_6,decay=dec,is_training=state)
    '''
    layer_6 = tf.layers.conv2d(layer_5,filters=256,kernel_size=3,strides=2,padding='SAME',activation=act_mode)
    layer_6 = tf.contrib.layers.batch_norm(layer_6,decay=dec,is_training=state)
    '''
    #print(layer_6.shape)
    
    #################block 4
    layer_7 = tf.layers.conv2d(layer_6,filters=512,kernel_size=3,strides=1,padding='SAME',activation=act_mode)
    layer_7 = tf.contrib.layers.batch_norm(layer_7,decay=dec,is_training=state)
    #if state is not None:
        #low_level2=layer_7
    #else:
    
        #print(layer_6.shape)
    
        #################block 4
   
    print(layer_7.shape)
    low_level1=layer_7
    low_level2=layer_7
    return low_level1,low_level2


# In[52]:


def midlevel_feature(lowlevel_layer1,dec,act_mode,state=True):
    layer_1 = tf.layers.conv2d(lowlevel_layer1,filters=512,kernel_size=3,strides=1,padding='SAME',activation=act_mode)
    layer_1 = tf.contrib.layers.batch_norm(layer_1,decay=dec,is_training=state)
    
    layer_2 = tf.layers.conv2d(layer_1,filters=256,kernel_size=3,strides=1,padding='SAME',activation=act_mode)
    layer_2 = tf.contrib.layers.batch_norm(layer_2,decay=dec,is_training=state)
    print(layer_2.shape)
    return layer_2


# In[53]:


def global_feature(lowlevel_layer2,dec,act_mode,state=True):
    layer_1 = tf.layers.conv2d(lowlevel_layer2,filters=512,kernel_size=3,strides=2,padding='SAME',activation=act_mode)
    layer_1 = tf.contrib.layers.batch_norm(layer_1,decay=dec,is_training=state)
    
    layer_2 = tf.layers.conv2d(layer_1,filters=512,kernel_size=3,strides=1,padding='SAME',activation=act_mode)
    layer_2 = tf.contrib.layers.batch_norm(layer_2,decay=dec,is_training=state)
    
    layer_3 = tf.layers.conv2d(layer_2,filters=512,kernel_size=3,strides=2,padding='SAME',activation=act_mode)
    layer_3 = tf.contrib.layers.batch_norm(layer_3,decay=dec,is_training=state)
    
    layer_4 = tf.layers.conv2d(layer_3,filters=512,kernel_size=3,strides=1,padding='SAME',activation=act_mode)
    layer_4 = tf.contrib.layers.batch_norm(layer_4,decay=dec,is_training=state)
    
    layer_4_flat = tf.reshape(layer_4,[-1,layer_4.shape[1]*layer_4.shape[2]*layer_4.shape[3]])
    
    layer_5 = tf.contrib.layers.fully_connected(layer_4_flat,num_outputs=1024)
    
    layer_6 = tf.contrib.layers.fully_connected(layer_5,num_outputs=512)
    
    layer_7 = tf.contrib.layers.fully_connected(layer_6,num_outputs=64)
    
    print(layer_7.shape)
    return layer_7


# In[54]:


def mini_auto_encode(lowlevel_layer2,dec,act_mode,state=True):
    layer_1 = tf.layers.conv2d(lowlevel_layer2,filters=256,kernel_size=11,strides=2,padding='SAME',activation=act_mode)
    layer_1 = tf.contrib.layers.batch_norm(layer_1,decay=dec,is_training=state)
    
    layer_2 = tf.layers.conv2d(layer_1,filters=256,kernel_size=9,strides=1,padding='SAME',activation=act_mode)
    layer_2 = tf.contrib.layers.batch_norm(layer_2,decay=dec,is_training=state)
    
    layer_3 = tf.layers.conv2d(layer_2,filters=128,kernel_size=7,strides=2,padding='SAME',activation=act_mode)
    layer_3 = tf.contrib.layers.batch_norm(layer_3,decay=dec,is_training=state)
    
    layer_4 = tf.layers.conv2d_transpose(layer_3,filters=128,kernel_size=7, strides=2,padding='SAME')
    
    layer_5 = tf.layers.conv2d_transpose(layer_4,filters=64,kernel_size=9, strides=2,padding='SAME')
    
    #layer_6 = tf.layers.conv2d_transpose(layer_5,filters=64,kernel_size=11, strides=2,padding='SAME')
    print(layer_5.shape)
    
    return layer_5


# In[58]:


##################global_layer should just be an vector with 256 dimension
def combine(mid_layer,global_layer,mini_auto):
    '''
    if mid_layer.shape[0]!=global_layer.shape[0] or mid_layer.shape[1]!=global_layer.shape[1] or mid_layer.shape[2]!=global_layer.shape[2]:
        print(mid_layer.shape[0],mid_layer.shape[1],mid_layer.shape[2])
        print(global_layer.shape[0],global_layer.shape[1],global_layer.shape[2])
    catted = tf.concat([mid_layer,global_layer], axis=3)
    '''
    #global_layer_flat = tf.reshape(global_layer,[-1,global_layer.shape[1]*global_layer.shape[2]*global_layer.shape[3]])
    global_layer_flat = tf.reshape(global_layer,[-1,1,1,global_layer.shape[1]])
    global_layer_tile = tf.tile(global_layer_flat,[1,mid_layer.shape[1],mid_layer.shape[2],1])
    catted = tf.concat([mid_layer,global_layer_tile], axis=3)
    catted = tf.concat([catted,mini_auto], axis=3)
    #output_layer = tf.reshape(layer_7_flat,[-1,28,28,16])
    
    print(catted.shape)
    return catted


# In[61]:


def decoder(com,dec,act_mode,state=True):
    ############################decoder part
    ###################block 8
    #layer_1 = tf.contrib.layers.fully_connected(mid_layer,num_outputs=128)
    #layer_1 = tf.contrib.layers.batch_norm(layer_1,decay=dec,is_training=state)
    
    layer_1 = tf.layers.conv2d_transpose(com,filters=64, kernel_size=3,strides=2,padding='SAME')
    layer_1 = tf.contrib.layers.batch_norm(layer_1,decay=dec,is_training=state)
    #print(layer_1.shape)
    #catted = combine(layer_1,global_layer,mini_auto)
    
    layer_2 = tf.layers.conv2d(layer_1,filters=128,kernel_size=3,strides=1,padding='SAME',activation=act_mode)
    layer_2 = tf.contrib.layers.batch_norm(layer_2,decay=dec,is_training=state)
    
    layer_3 = tf.layers.conv2d(layer_2,filters=128,kernel_size=3,strides=1,padding='SAME',activation=act_mode)
    layer_3 = tf.contrib.layers.batch_norm(layer_3,decay=dec,is_training=state)
    
    
    layer_4 = tf.layers.conv2d(layer_3,filters=64,kernel_size=3,strides=1,padding='SAME',activation=act_mode)
    layer_4 = tf.contrib.layers.batch_norm(layer_4,decay=dec,is_training=state)
    
    layer_5 = tf.layers.conv2d(layer_4,filters=64,kernel_size=3,strides=1,padding='SAME',activation=act_mode)
    layer_5 = tf.contrib.layers.batch_norm(layer_5,decay=dec,is_training=state)
    
    #print layer_7.shape
    
    ##########if we use nn we need to change the shape of the previous layer
    
    layer_6 = tf.layers.conv2d_transpose(layer_5,filters=64,kernel_size=3, strides=2,padding='SAME')
    
    layer_7 = tf.layers.conv2d(layer_6,filters=64,kernel_size=3,strides=1,padding='SAME',activation=tf.nn.sigmoid)
    layer_7 = tf.contrib.layers.batch_norm(layer_7,decay=dec,is_training=state)
    
    layer_8 = tf.layers.conv2d_transpose(layer_7,filters=2,kernel_size=3, strides=2,padding='SAME')
    
    ##################################################
    output = layer_8
    #output = tf.contrib.layers.batch_norm(output,decay=dec,is_training=state)
    print (output.shape)
    return output


# In[62]:


#import net

######define input
X = tf.placeholder("float", [None,224,224,1])#gray_image
R = tf.placeholder("float", [None,224,224,1])#R_image
G = tf.placeholder("float", [None,224,224,1])#G_image
B = tf.placeholder("float", [None,224,224,1])#B_image
output = tf.placeholder("float", [None,224,224,1]) #the ab image
output2 = tf.placeholder("float", [None,224,224,2]) #use 2 dimension ab image
l_r = tf.placeholder("float",None)
state = tf.placeholder("bool",None)

dec=0.9
#state=True
'''
#output_r=net.architecture(X, train=True)
#output_g=net.architecture(X, train=True)
#output_b=net.architecture(X, train=True)

r_layer_8 = architecture1(X, 0.9,tf.nn.relu,state)
g_layer_8 = architecture2(X, 0.9,tf.nn.relu,state)
b_layer_8 = architecture3(X, 0.9,tf.nn.relu,state)

#layer_8=generate_image(r_layer_8,g_layer_8,b_layer_8)
#print layer_8.shape
loss_r= redefine_loss(r_layer_8, R)
loss_g = redefine_loss(g_layer_8,G)
loss_b = redefine_loss(b_layer_8,B)
#loss_r = tf.reduce_mean(tf.nn.l2_loss(r_layer_8-R))
#loss_g = tf.reduce_mean(tf.nn.l2_loss(g_layer_8-G))
#loss_b = tf.reduce_mean(tf.nn.l2_loss(b_layer_8-B))


train_op_r = tf.train.AdamOptimizer(learning_rate=l_r).minimize(loss_r)#, global_step=global_step)
train_op_g = tf.train.AdamOptimizer(learning_rate=l_r).minimize(loss_g)#, global_step=global_step)
train_op_b = tf.train.AdamOptimizer(learning_rate=l_r).minimize(loss_b)#, global_step=global_step)
'''
#####################for ab image
act_mode=tf.nn.relu
lowlevel_layer1,lowlevel_layer2 = encoder(X,X,dec,act_mode,state)
mid_layer = midlevel_feature(lowlevel_layer1,dec,act_mode,state)
global_layer = global_feature(lowlevel_layer2,dec,act_mode,state)
auto_en = mini_auto_encode(lowlevel_layer2,dec,act_mode,state)
com_layer = combine(mid_layer,global_layer,auto_en)
output_layer = decoder(com_layer,dec,act_mode,state)

loss = redefine_loss(output_layer,output2)
train_op = tf.train.AdamOptimizer(learning_rate=l_r).minimize(loss)#, global_step=global_step)


# In[63]:


def show_test_image(test_image,target_image,sess):
    #test_image = gray_image[0]
    i_img = np.reshape(test_image,[1,224,224,1])
    o_img = np.reshape(target_image,[1,224,224,2])
    #print t_image.shape
    #pred = sess.run([layer_8], feed_dict={X: t_image,state:True})
    #######################when we have three loss
    '''
    pred_r,pred_g,pred_b = sess.run([r_layer_8,g_layer_8,b_layer_8], feed_dict={X: t_image,state:True})
    
    pred_r = np.array(pred_r)
    pred_g = np.array(pred_g)
    pred_b = np.array(pred_b)
    #pred = np.reshape(pred,[pred.shape[-3],pred.shape[-2],3])
    
    image = np_generate_image(pred_r,pred_g,pred_b)
    '''
    pre_img = sess.run([output_layer], feed_dict={X:i_img,state:True})
    pre_img = np.reshape(pre_img,[224,224,2])
    #print image
    plt.figure(figsize=(10,10))
    plt.subplot(311)
    plt.imshow(test_image,cmap='gray')
    '''    
    plt.subplot(152)
    plt.imshow(target_image[:,:,0])
        
    plt.subplot(153)
    plt.imshow(target_image[:,:,1])
    '''
    L = np.reshape(test_image,[224,224])
    #################get the target output
    a = np.reshape(target_image[:,:,0],[224,224])
    b = np.reshape(target_image[:,:,1],[224,224])
    transfer = cv2.merge([L,a,b])
    trans = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2RGB)
    img = Image.fromarray(trans, 'RGB')
    plt.subplot(312)
    plt.imshow(img)
    #print pre_img
    #print target_image
    #print pre_img-target_image
    #################get the predict output
    
    a = np.reshape(pre_img[:,:,0],[224,224])
    b = np.reshape(pre_img[:,:,1],[224,224])
    transfer = cv2.merge([L,a,b])
    trans = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2RGB)
    img = Image.fromarray(trans, 'RGB')
    plt.subplot(313)
    plt.imshow(img)
    plt.show()
    '''
    plt.subplot(154)
    plt.imshow()
    
    plt.subplot(155)
    plt.imshow()
    '''
    plt.show()


# In[64]:


import time
import random

'''
input is:
X:gray_image=np.array(gray_image)
or 
l_img
output(target):
image 
or
R:r_image=np.array(image)[:,:,:,0]
G:g_image=np.array(image)[:,:,:,1]
B:b_image=np.array(image)[:,:,:,2]
or use the uint8 file 
o_img
or lab ab image 
lab2color_label
or
lab_label_2 ##2 dimension ab image
'''
training_epochs=1000
saver = tf.train.Saver()
batchsize=32
with tf.Session() as sess:
    #################if this is the first time train
    tf.global_variables_initializer().run(session=sess)
    ################if you need the previous model variable
    #saver.restore(sess, "./CSE253model2.ckpt")
    # Training cycle
    #L=Loader(img_path) ###default 
    learning=0.01
    total_loss=[]
    for epoch in range(training_epochs):
        Loss=0
        start_time = time.time()
        #print(L.n_batches()[0])
        #target_img=np.array(lab2color_label)
        np.random.shuffle(lab_all)
        #train_set=lab_all[:len(lab_all)*0.8]
        l_img,lab_label_2 = get_i_o(lab_all)     #it contanin the shuffle part
        target_img = np.array(lab_label_2)
        input_image = np.array(l_img)
        for s in range(int(len(input_image)/batchsize)):#range(100):
            
            if (s+1)*batchsize>=len(input_image):
                index=len(input_image)
            else:
                index=(s+1)*batchsize
            
            #input_image=gray_image[s*batchsize:index]
            #R_color=target_img[s*batchsize:index]
            
            #Get input part
            
            i_img=input_image[s*batchsize:index]
            o_img=target_img[s*batchsize:index]
            '''
            #########only use the first picture train
            i_img=input_image[177]
            o_img=target_img[177]
            '''
            #R_color=r_image[0]
            #G_color=g_image[0]
            #B_color=b_image[0]
            #########reshape image and color
            i_img = np.reshape(i_img,[len(i_img),224,224,1])
            #print R_color.shape
            #############reshape label part
            o_img = np.reshape(o_img,[len(o_img),224,224,2])
            #R_color = np.reshape(R_color,[batchsize,R_color.shape[0],R_color.shape[1],1])
            #G_color = np.reshape(G_color,[batchsize,G_color.shape[0],G_color.shape[1],1])
            #B_color = np.reshape(B_color,[batchsize,B_color.shape[0],B_color.shape[1],1])
            #print (R_color.shape)
            #print('begin run')
            _, lo = sess.run([train_op, loss], feed_dict={X:i_img ,output2: o_img,l_r:learning,state:True})
            '''
            _, loss_red = sess.run([train_op_r, loss_r], feed_dict={X:input_image ,R: R_color,l_r:learning,state:True})
                                                           #keep_conv: 0.1, keep_hidden: 0.1,learning_rate:L_r})
            '''
            '''
            _,_,_, loss_red,loss_green,loss_blue = sess.run([train_op_r,train_op_g,train_op_b, loss_r,loss_g,loss_b], 
                                   feed_dict={X:input_image ,R: R_color,G:G_color,B:B_color,l_r:learning,state:True})
            '''
            #print(np.array(pred).shape)
            '''
            L_r=loss_red/int(len(gray_image)/batchsize)
            L_g=loss_green/int(len(gray_image)/batchsize)
            L_b=loss_blue/int(len(gray_image)/batchsize)
            Loss+=((L_r+L_g+L_b)/3)
            '''
            #Loss+=(lo/int(len(input_image)/batchsize))
            ##########whenwe only use q image to train
            Loss+=(lo/100)
            #print(lo)
        #if epoch%5==0:
            #save_path = saver.save(sess, "./CSE291FinalModel.ckpt")  #save model for second part
        #pred = sess.run([logits], feed_dict={images: [image[0]]})
        #plt.imshow(np.reshape(image[0],[304,228,3]))
        #plt.show()
        #plt.imshow(np.reshape(pred,[74,55]))
        #plt.show() 
        ###########see the image
        print("--- %s seconds ---" % (time.time() - start_time))
        print(Loss)
        total_loss.append(Loss)
        
        if epoch%10==0:
            #################plot test image
            index = random.randint(0, len(input_image)-1)
            in_img=input_image[index]
            out_img=target_img[index]
            show_test_image(in_img,out_img,sess)
            
        if epoch%20==0 and epoch!=0:
            save_path = saver.save(sess, "./CSE253model_Mountain_change.ckpt")
            learning/=2.
        

