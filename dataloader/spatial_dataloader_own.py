import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import os

from split_train_test_video_own import *

from skimage import io, color, exposure

class spatial_dataset_own(Dataset):
    def __init__(self, dic, root_dir, mode, transform=None):
 
        self.keys = dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.mode =mode
        self.transform = transform
        self.pic_dir="/nfs/syzhou/github/two-stream-action-recognition/UCF-101-pic_own/"

    def __len__(self):
        return len(self.keys)

    def load_ucf_image(self,video_name, index):

        path=self.pic_dir+video_name
        #print(index)
        #print(path)
        img = Image.open(path +'/'+str(index)+'.jpeg')
        transformed_img = self.transform(img)
        img.close()

        return transformed_img

    def __getitem__(self, idx):

        if self.mode == 'train':
            self.keys=list(self.keys)
            #print(self.keys[idx])
            video_name, nb_clips = self.keys[idx].split(' ')
            #print(self.keys[idx])
            #video_name, nb_clips = idx_keys.split(' ')
            nb_clips = int(nb_clips)
            clips = []
            clips.append(random.randint(1, int(nb_clips/3)))
            clips.append(random.randint(int(nb_clips/3), int(nb_clips*2/3)))
            clips.append(random.randint(int(nb_clips*2/3), int(nb_clips+1)))
            
        elif self.mode == 'val':
            self.keys=list(self.keys)
            video_name, index = self.keys[idx].split(' ')
            index =abs(int(index))
        else:
            raise ValueError('There are only train and val mode')
        self.values=list(self.values) 
        label = self.values[idx]
        label = int(label)-1
        
        if self.mode=='train':
            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                data[key] = self.load_ucf_image(video_name, index)
                    
            sample = (data, label)
        elif self.mode=='val':
            data = self.load_ucf_image(video_name,index)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class spatial_dataloader_own():
    def __init__(self, BATCH_SIZE, num_workers, path, ucf_list, ucf_split):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.frame_count ={}
        # split the training and testing videos
        splitter = UCF101_splitter_own(path=ucf_list,split=ucf_split)
        self.train_video, self.test_video = splitter.split_video()

    def generate_frame_count(self):
        data_pic_path = '/nfs/syzhou/github/two-stream-action-recognition/UCF-101-pic_own/'
        all_pics_dir = os.listdir(data_pic_path);
        dic = {};
        num = 0
        for curr_category in all_pics_dir:
            num = num + 1
            curr_dir = os.path.join(data_pic_path, curr_category)
            # print(data_pic_path,curr_category)
            # pictures=['jpeg','jpg','bmp','png']
            for pictures in os.listdir(curr_dir):
                num_frames = len(os.listdir(os.path.join(curr_dir)))
                # print(num_frames)
                # print(pictures) 73.jpeg
                # dic[pictures] = num_frames
            # dic [path_to_training_frames] = video_label
            curr_category = curr_category + '.avi'
            dic[curr_category] = num_frames
            str_write = curr_category + '\n' + str(num) + '\n' + str(num_frames)
            #print(str_write)
            # pickle.dump(str_write, open('/nfs/syzhou/github/two-stream-action-recognition/dataloader/dic_own/frame_count.pickle','wb'))
            # jushou_1.avi
            # dic[curr_category]=num_frames
        #print(dic)
        pickle.dump(dic, open('/nfs/syzhou/github/two-stream-action-recognition/dataloader/dic_own/frame_count.pickle',
                              'wb'))

    def load_frame_count(self):
        #print '==> Loading frame number of each video'
        pickle_path = '/nfs/syzhou/github/two-stream-action-recognition/dataloader/dic_own/frame_count.pickle'
        if not os.path.exists(pickle_path):
            self.generate_frame_count()
        with open('/nfs/syzhou/github/two-stream-action-recognition/dataloader/dic_own/frame_count.pickle','rb') as file:
            dic_frame = pickle.load(file)
        file.close()
        for line in dic_frame :
            videoname = line.split('.',1)[0]
            self.frame_count[videoname]=dic_frame[line]


    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample20()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    def get_training_dic(self):
        print ('==> Generate frame numbers of each training video')
        self.dic_training={}
        for video in self.train_video:
            #print (videoname)
            nb_frame = self.frame_count[video]-10+1
            key = video+' '+ str(nb_frame)
            self.dic_training[key] = self.train_video[video]
                    
    def val_sample20(self):
        print ('==> sampling testing frames')
        self.dic_testing={}
        for video in self.test_video:
            nb_frame = self.frame_count[video]-10+1
            interval = int(nb_frame/19)
            for i in range(19):
                frame = i*interval
                key = video+ ' '+str(frame+1)
                self.dic_testing[key] = self.test_video[video]      

    def train(self):
        training_set = spatial_dataset_own(dic=self.dic_training, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print ('==> Training data :',len(training_set),'frames')
        training_set=list(training_set)
        print (training_set[1][0]['img1'].size())

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        validation_set = spatial_dataset_own(dic=self.dic_testing, root_dir=self.data_path, mode='val', transform = transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print ('==> Validation data :',len(validation_set),'frames')
        print (validation_set[1][1].size())

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader


if __name__ == '__main__':
    
    dataloader = spatial_dataloader_own(BATCH_SIZE=1, num_workers=1,
                                path='/nfs/syzhou/github/two-stream-action-recognition/UCF101_own/',
                                ucf_list='/nfs/syzhou/github/two-stream-action-recognition/UCF_list_own/',
                                ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()