import numpy as np
import pickle
from PIL import Image
import time
import shutil
import random
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
print(sys.path)

from split_train_test_video_own import *
#import split_train_test_video
 
class motion_dataset(Dataset):  
    def __init__(self, dic, in_channel, root_dir, mode, transform=None):
        #Generate a 16 Frame clip
        self.keys=dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.transform = transform
        self.mode=mode
        self.in_channel = in_channel
        self.img_rows=224
        self.img_cols=224

    def stackopf(self):
        name = self.video+'.avi'
        u = self.root_dir+ 'u/' + name
        v = self.root_dir+ 'v/'+ name
        
        flow = torch.FloatTensor(2*self.in_channel,self.img_rows,self.img_cols)
        i = int(self.clips_idx)


        for j in range(self.in_channel):
            #/tvl1_flow_own/u/v_handsup_1/frame000174.jpg
            #/tvl1_flow_own/u/dancing_1.avi/flow_x_00096.jpg

            idx = i + j
            idx = str(idx)
            #print(idx)
            frame_idx =  idx.zfill(5)
            h_image = u +'/' + 'flow_x_'+frame_idx +'.jpg'
            v_image = v +'/' + 'flow_y_'+frame_idx +'.jpg'
            
            imgH=(Image.open(h_image))
            imgV=(Image.open(v_image))

            H = self.transform(imgH)
            V = self.transform(imgV)

            
            flow[2*(j-1),:,:] = H
            flow[2*(j-1)+1,:,:] = V      
            imgH.close()
            imgV.close()  
        return flow

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        
        if self.mode == 'train':
            self.keys=list(self.keys)
            self.video, nb_clips = self.keys[idx].split('-')
            self.clips_idx = random.randint(1,int(nb_clips))
        elif self.mode == 'val':
            self.keys=list(self.keys)
            self.video,self.clips_idx = self.keys[idx].split('-')
        else:
            raise ValueError('There are only train and val mode')

        self.values=list(self.values)
        label = self.values[idx]
        label = int(label)-1 
        data = self.stackopf()

        if self.mode == 'train':
            sample = (data,label)
        elif self.mode == 'val':
            sample = (self.video,data,label)
        else:
            raise ValueError('There are only train and val mode')
        return sample





class Motion_DataLoader_own():
    def __init__(self, BATCH_SIZE, num_workers, in_channel,  path, ucf_list, ucf_split):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.frame_count={}
        self.in_channel = in_channel
        self.data_path=path
        # split the training and testing videos
        splitter = UCF101_splitter_own(path=ucf_list,split=ucf_split)
        self.train_video, self.test_video = splitter.split_video()
        
    def load_frame_count(self):
        #print '==> Loading frame number of each video'
        with open('/nfs/syzhou/github/two-stream-action-recognition/dataloader/dic_own/frame_count.pickle','rb') as file:
            dic_frame = pickle.load(file)
        file.close()

        for line in dic_frame :
            videoname = line.split('.',1)[0]
            #print(videoname)
            self.frame_count[videoname]=dic_frame[line] 

    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample19()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader, self.test_video
            
    def val_sample19(self):
        self.dic_test_idx = {}
        #print len(self.test_video)
        for video in self.test_video:

            sampling_interval = int((self.frame_count[video]-10+1)/19)
            for index in range(19):
                clip_idx = index*sampling_interval
                key = video + '-' + str(clip_idx+1)
                self.dic_test_idx[key] = self.test_video[video]
             
    def get_training_dic(self):
        self.dic_video_train={}
        for video in self.train_video:
            print(video)
            nb_clips = self.frame_count[video]-10+1
            key = video +'-' + str(nb_clips)
            self.dic_video_train[key] = self.train_video[video] 
                            
    def train(self):
        training_set = motion_dataset(dic=self.dic_video_train, in_channel=self.in_channel, root_dir=self.data_path,
            mode='train',
            transform = transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            ]))
        print ('==> Training data :',len(training_set),' videos',training_set[1][0].size())

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
            )

        return train_loader

    def val(self):
        validation_set = motion_dataset(dic= self.dic_test_idx, in_channel=self.in_channel, root_dir=self.data_path ,
            mode ='val',
            transform = transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            ]))
        print ('==> Validation data :',len(validation_set),' frames',validation_set[1][1].size())
        #print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader

if __name__ == '__main__':
    data_loader =Motion_DataLoader_own(BATCH_SIZE=1,num_workers=1,in_channel=10,
                                        path='/nfs/syzhou/github/two-stream-action-recognition/tvl1_flow_own/',
                                        ucf_list='/nfs/syzhou/github/two-stream-action-recognition/UCF_list_own/',
                                        ucf_split='01'
                                        )
    train_loader,val_loader,test_video = data_loader.run()
    #print train_loader,val_loader