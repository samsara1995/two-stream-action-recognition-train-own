from matplotlib import pyplot as plt

import os,sys
import numpy as np
import cv2
#import cv2
from PIL import Image
from multiprocessing import Pool
import argparse
#from IPython import embed #to debug
import skvideo.io
import scipy.misc

def Basic_operation():
    data_path='/nfs/syzhou/github/two-stream-action-recognition/tvl1_flow_own_2/'
    save_path='/nfs/syzhou/github/two-stream-action-recognition/tvl1_flow_own_er/'
    u_path=data_path+'u/'
    u_path_save=save_path+'u/'
    v_path_save=save_path+'v/'
    v_path=data_path+'v/'
    all_data_path = os.listdir(u_path)

    dict_path = {}
    for files in all_data_path:
        dir_path=u_path+files+'/'
        all_data_path_1=os.listdir(dir_path)
        num=0
        for files_1 in all_data_path_1:
            num=num+1
        dict_path[files]=num
    print(dict_path)#{'shuijiao_1.avi': 42, 'dancing_1.avi': 27, 'handsup_1.avi': 45, 'jushou_1.avi': 100, 'interrupt_1.avi': 120}


    for files in all_data_path:
        #print(files)
        dir_path=u_path+files+'/'
        dir_path_save=u_path_save+files+'/'
        all_data_path_1=os.listdir(dir_path)
        for files_1 in all_data_path_1:
            pic_path_save=dir_path_save+files_1
            pic_path=dir_path+files_1
            src = cv2.imread(pic_path)
            #Binarization_pic()
            dst=Binarization_pic(src)

            flag=Histogram_variance(dst)
            if flag==1 :
                print('yes',dir_path,files_1)
                Calculated_position_u(dir_path,files_1,dict_path)
            if not os.path.exists(os.path.join(dir_path_save)):
                os.makedirs(os.path.join(dir_path_save))
            cv2.imwrite(pic_path_save, dst)

    all_data_path=os.listdir(v_path)
    for files in all_data_path:
        dir_path=v_path+files+'/'
        dir_path_save=v_path_save+files+'/'
        all_data_path_1=os.listdir(dir_path)
        for files in all_data_path_1:

            pic_path_save = dir_path_save+files
            pic_path=dir_path+files
            src = cv2.imread(pic_path)
            dst=Binarization_pic(src)
            flag = Histogram_variance(dst)
            if flag==1 :
                print('yes',dir_path,files)
                pic_rate =Calculated_position_v(dir_path, files,dict_path)
            #elif flag==0:
            #    print('no',dir_path,files)
            if not os.path.exists(os.path.join(dir_path_save)):
                os.makedirs(os.path.join(dir_path_save))
            cv2.imwrite(pic_path_save, dst)

def Binarization_pic(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    a,dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    #print(a) a 为阈值
    return dst


def split_vedio():
    data_pic_path = '/nfs/syzhou/github/two-stream-action-recognition/UCF-101-pic_own_2/'
    data_path = '/nfs/syzhou/github/two-stream-action-recognition/UCF-101_own/'
    all_data_path = os.listdir(data_path)

    for files in all_data_path:
        data_path_1 = data_path + files + '/'
        all_data_path_1 = os.listdir(data_path_1)
        for files_1 in all_data_path_1:
            print(data_path_1)
            print(files_1)
            str1 = data_path_1 + files_1
            name = files_1.split('.')[0]
            print(name)
            if is_have_dir(data_pic_path, name) == False:
                makdir(data_pic_path, name)
            chuanru = "ffmpeg -i " + str1 + ' -r 1'+" -f image2 /nfs/syzhou/github/two-stream-action-recognition/UCF-101-pic_own_2/" + name + "/%d.jpeg -q:v 2 "
            print(chuanru)

            # ffmpeg -i /nfs/syzhou/github/two-stream-action-recognition/UCF-101_own/jushou/jushou_1.avi /nfs/syzhou/github/two-stream-action-recognition/UCF-101-pic_own/jushou_1/%d.jpeg -r 0.5 -q:v 2
            # ffmpeg -i /nfs/syzhou/github/two-stream-action-recognition/UCF-101_own/shuijiao/shuijiao_1.avi  /nfs/syzhou/github/two-stream-action-recognition/UCF-101-pic_own/2/%d.jpeg -r 0.5 -q:v 2 "
            c = os.popen(chuanru)  # 用于从一个命令打开一个管道。
            # 将名为*.mp4的视频文件抽成一张张的图片（抽帧）
            # ffmpeg -i "*.mp4" -r 1 -q:v 2 -f image2 %d.jpeg
            # -i 是用来获取输入的文件，-i “ *.mp4” 就是获取这个叫做星号的mp4视频文件；
            # -r 是设置每秒提取图片的帧数，-r 1 的意思就是设置为每秒获取一帧；
            # -q: v2 这个据说是提高抽取到的图片的质量的，具体的也没懂；
            # -f   据说是强迫采用格式fmt
            out = c.read()

def is_have_dir(data_pic_path, name):
        tobecheckdir = data_pic_path + name + '/'
        print(tobecheckdir)
        return os.path.isdir(tobecheckdir)


def makdir(data_pic_path, name):
    path = data_pic_path + name + '/'
    os.mkdir(path)
    print('make dir', name)
'''
def Expansion_corrosion():
'''
def Histogram_variance(dst):
    hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
    if hist[0] < hist[len(hist) - 1]:
        small_part = hist[0]
    else:
        small_part = hist[len(hist) - 1]
    sum_pixal = int(hist[0]) + int(hist[len(hist) - 1])
    Threshold = sum_pixal / 3
    Threshold_1 = sum_pixal / 20
    flag = 0
    if small_part > Threshold or small_part < Threshold_1:
        flag = 0
        # 去除这个图片
    elif (small_part < Threshold) and (small_part > Threshold_1):
        flag = 1
    #print(small_part)
    return flag

def Calculated_position_u(dir_path,files,dict_path):
    #/nfs/syzhou/github/two-stream-action-recognition/tvl1_flow_own_2/u/dancing_1.avi/ flow_x_00012.jpg
    video_name = dir_path.split('.')[0].split('/u/')[1]
    pic_name=files.split('.')[0].split('x_')[1]
    print(video_name,pic_name)
    #d = {key1 : value1, key2 : value2 }
#    whole_frame=dict.get(key, default=None)
    for key in dict_path:
        #print(key,dict_path[key])
        if video_name in key:
            #print(dict_path[key])
            whole_frame=dict_path[key]
    print(whole_frame)
    pic_name=int(str(pic_name))
    pic_rate=pic_name/whole_frame
    print(pic_rate)
    return pic_rate

def Calculated_position_v(dir_path,files,dict_path):
    #/nfs/syzhou/github/two-stream-action-recognition/tvl1_flow_own_2/u/dancing_1.avi/ flow_x_00012.jpg
    video_name = dir_path.split('.')[0].split('/v/')[1]
    pic_name=files.split('.')[0].split('y_')[1]
    print(video_name,pic_name)
    for key in dict_path:
        #print(key,dict_path[key])
        if video_name in key:
            #print(dict_path[key])
            whole_frame=dict_path[key]
    print(whole_frame)
    pic_name=int(str(pic_name))
    pic_rate=pic_name/whole_frame
    print(pic_rate)
    return pic_rate


if __name__ == '__main__':
    #split_vedio()#切割视频
    #提取光流
    Basic_operation()
