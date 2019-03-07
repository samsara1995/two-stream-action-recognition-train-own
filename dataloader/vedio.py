

import os,sys



def VideoFraming(video_path):
    #a, b, c = os.popen("ffmpeg -i {}".format(video_path))
    a, b, c = os.popen("ffmpeg -i /nfs/syzhou/github/two-stream-action-recognition/UCF-101_own/shuijiao/shuijiao_1.avi  /nfs/syzhou/github/two-stream-action-recognition/UCF-101-pic_own/2/%d.jpeg -r 0.5 -q:v 2 ")#用于从一个命令打开一个管道。
    # 将名为*.mp4的视频文件抽成一张张的图片（抽帧）
    #ffmpeg -i "*.mp4" -r 1 -q:v 2 -f image2 %d.jpeg
    #-i 是用来获取输入的文件，-i “ *.mp4” 就是获取这个叫做星号的mp4视频文件；
    #-r 是设置每秒提取图片的帧数，-r 1 的意思就是设置为每秒获取一帧；
    #-q: v2 这个据说是提高抽取到的图片的质量的，具体的也没懂；
    #-f   据说是强迫采用格式fmt

    out = c.read()
    dp = out.index("Duration: ")
    duration = out[dp+10:dp+out[dp:].index(",")]
    hh, mm, ss = map(float, duration.split(":"))
    #total time ss
    total = (hh*60 + mm)*60 + ss
    for i in range(9):
        t = int((i + 1) * total / 10)
        # ffmpeg -i test.mp4 -y -f mjpeg -ss 3 -t 1  test1.jpg
        os.system("ffmpeg -i %s -y -f mjpeg -ss %s -t 1 img/img_%i.jpg" % (video_path,t, i))
    return True

vedio_path='/nfs/tmp/pycharm_project_350/10_Stu_LookingAround.avi'
print(VideoFraming(vedio_path))