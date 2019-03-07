import os
import pickle

'''
根据ucf-101-own里的视频，生成classind.txt和testlist01.txt和trainlist01.txt
根据UCF-101-own里的视频自动分割图片，保存在UCF-101-pic-own里

'''

def write_txt(data_path,txt_path):

        classInd_path = txt_path + 'classInd.txt'
        f1=open(classInd_path,'w')
        trainlist_path = txt_path + 'trainlist01.txt'
        testlist_path = txt_path + 'testlist01.txt'
        f2 = open(testlist_path, 'w')
        f3 = open(trainlist_path, 'w')

        all_data_path=os.listdir(data_path)
        for files in all_data_path:
                print(files)
                write_file=files+'\n'
                f1=open(classInd_path,'r+')
                f1.read()
                f1.write(write_file)
                f1.close()
                data_path_1=data_path+files+'/'
                all_data_path_1=os.listdir(data_path_1)
                for files_1 in all_data_path_1:
                        write_str=files+'/'+files_1+'\n'
                        print(write_str)
                        f2=open(testlist_path,'r+')
                        f2.read()
                        f2.write(write_str)
                        f2.close()
                        f3=open(trainlist_path,'r+')
                        f3.read()
                        f3.write(write_str)
                        f3.close()
                        # jushou/jushou_1.avi
                        # shuijiao/shuijiao_1.avi


def split_vedio(data_path,data_pic_path):
        all_data_path=os.listdir(data_path)
        for files in all_data_path:
                data_path_1=data_path+files+'/'
                all_data_path_1=os.listdir(data_path_1)
                for files_1 in all_data_path_1:
                        print(data_path_1)
                        print(files_1)
                        str1=data_path_1+files_1
                        name=files_1.split('.')[0]
                        print(name)
                        if is_have_dir(data_pic_path,name)==False:
                                makdir(data_pic_path,name)
                        chuanru="ffmpeg -i "+str1+" /nfs/syzhou/github/two-stream-action-recognition/UCF-101-pic_own/"+name+"/%d.jpeg -r 0.5 -q:v 2 "
                        print(chuanru)
                        #ffmpeg -i /nfs/syzhou/github/two-stream-action-recognition/UCF-101_own/jushou/jushou_1.avi /nfs/syzhou/github/two-stream-action-recognition/UCF-101-pic_own/jushou_1/%d.jpeg -r 0.5 -q:v 2
                        #ffmpeg -i /nfs/syzhou/github/two-stream-action-recognition/UCF-101_own/shuijiao/shuijiao_1.avi  /nfs/syzhou/github/two-stream-action-recognition/UCF-101-pic_own/2/%d.jpeg -r 0.5 -q:v 2 "
                        c=os.popen(chuanru)  # 用于从一个命令打开一个管道。
                        # 将名为*.mp4的视频文件抽成一张张的图片（抽帧）
                        # ffmpeg -i "*.mp4" -r 1 -q:v 2 -f image2 %d.jpeg
                        # -i 是用来获取输入的文件，-i “ *.mp4” 就是获取这个叫做星号的mp4视频文件；
                        # -r 是设置每秒提取图片的帧数，-r 1 的意思就是设置为每秒获取一帧；
                        # -q: v2 这个据说是提高抽取到的图片的质量的，具体的也没懂；
                        # -f   据说是强迫采用格式fmt
                        out = c.read()
                        '''
                        dp = out.index("Duration: ")
                        duration = out[dp + 10:dp + out[dp:].index(",")]
                        hh, mm, ss = map(float, duration.split(":"))
                        # total time ss
                        total = (hh * 60 + mm) * 60 + ss
                        for i in range(9):
                                t = int((i + 1) * total / 10)
                                # ffmpeg -i test.mp4 -y -f mjpeg -ss 3 -t 1  test1.jpg
                                os.system("ffmpeg -i %s -y -f mjpeg -ss %s -t 1 img/img_%i.jpg" % (video_path, t, i))
                        '''

def is_have_dir(data_pic_path,name):
        tobecheckdir = data_pic_path+name+'/'
        print(tobecheckdir)
        return os.path.isdir(tobecheckdir)

def makdir(data_pic_path,name):
        path=data_pic_path+name+'/'
        os.mkdir(path)
        print('make dir',name)
'''
根据UCF-101_own里的分割好的图片，生成frame_count.pickle文件.
dic [path_to_training_frames] = video_label
'''
def create_pickle(data_pic_path):
        all_pics_dir = os.listdir(data_pic_path);

        dic = {};
        num=0
        for curr_category in all_pics_dir:
                num=num+1
                # curr_dir_1='/nfs/syzhou/github/two-stream-action-recognition/UCF-101-pic_own/'
                curr_dir = os.path.join(data_pic_path, curr_category)
                #print(data_pic_path,curr_category)
                # if filepath3.endswith(fileType):

                # pictures=['jpeg','jpg','bmp','png']
                for pictures in os.listdir(curr_dir):
                        num_frames = len(os.listdir(os.path.join(curr_dir)))
                        #print(num_frames)
                        #print(pictures) 73.jpeg
                        #dic[pictures] = num_frames
                    #dic [path_to_training_frames] = video_label
                curr_category=curr_category+'.avi'
                #print(curr_category,num,num_frames)
                dic[curr_category]=num_frames
                str_write=curr_category+'\n'+str(num)+'\n'+str(num_frames)
                print(str_write)
                #jushou_1.avi
                #dic[curr_category]=num_frames
        print(dic)
        pickle.dump(dic, open('/nfs/syzhou/github/two-stream-action-recognition/dataloader/dic_own/frame_count.pickle','wb'))


if __name__ == '__main__':
        data_pic_path = '/nfs/syzhou/github/two-stream-action-recognition/UCF-101-pic_own/'
        data_path = '/nfs/syzhou/github/two-stream-action-recognition/UCF-101_own/'
        txt_path = '/nfs/syzhou/github/two-stream-action-recognition/UCF_list_own/'
        #write_txt(data_path, txt_path)
        #create_pickle(data_pic_path)
        split_vedio(data_path, data_pic_path)
