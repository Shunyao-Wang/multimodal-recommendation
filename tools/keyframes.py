import video.keyframes_extract as kf
import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
import random

# 图像裁剪为112*112，张数只保存16

DATAPATH = "../data/"
# DATAPATH = "F:/data/"
# DATAPATH = "/home/wang/dev/data/" # ubuntu

file = open(DATAPATH + 'movieID.txt', 'r')
movieID = [int(x) for x in file]
file.close()
print(movieID)


def kf_do(id):
    print(sys.executable)
    # Setting fixed threshold criteria
    USE_THRESH = False
    # fixed threshold value
    THRESH = 0.6
    # Setting fixed threshold criteria
    USE_TOP_ORDER = False
    # Setting local maxima criteria
    USE_LOCAL_MAXIMA = True
    # Number of top sorted frames
    NUM_TOP_FRAMES = 50

    # Video path of the source file
    videopath = DATAPATH + "videos/" + str(id) + "_video.mp4"
    # Directory to store the processed frames
    dir = DATAPATH + "extract_result/"
    # smoothing window size
    len_window = int(50)

    print("target video :" + videopath)
    print("frame save directory: " + dir)
    # load video and compute diff between frames
    cap = cv2.VideoCapture(str(videopath))
    curr_frame = None
    prev_frame = None
    frame_diffs = []
    frames = []
    success, frame = cap.read()
    i = 0
    while (success):
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        if curr_frame is not None and prev_frame is not None:
            # logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = kf.Frame(i, diff_sum_mean)
            frames.append(frame)
        prev_frame = curr_frame
        i = i + 1
        success, frame = cap.read()
    cap.release()

    # compute keyframe
    keyframe_id_set = set()
    if USE_TOP_ORDER:
        # sort the list in descending order
        frames.sort(key=operator.attrgetter("diff"), reverse=True)
        for keyframe in frames[:NUM_TOP_FRAMES]:
            keyframe_id_set.add(keyframe.id)
    if USE_THRESH:
        print("Using Threshold")
        for i in range(1, len(frames)):
            if (kf.rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff)) >= THRESH):
                keyframe_id_set.add(frames[i].id)
    if USE_LOCAL_MAXIMA:
        print("Using Local Maxima")
        diff_array = np.array(frame_diffs)
        sm_diff_array = kf.smooth(diff_array, len_window)
        frame_indexes = np.asarray(kf.argrelextrema(sm_diff_array, np.greater))[0]
        for i in frame_indexes:
            keyframe_id_set.add(frames[i - 1].id)

        plt.figure(figsize=(40, 20))
        plt.locator_params(numticks=100)
        plt.stem(sm_diff_array)
        plt.savefig(dir + 'plot.png')

    # "keyframe_id_set"保存关键帧的帧数，如果>16,随机选取16帧；如果<16，再随机生成几个直到满足条件
    if len(keyframe_id_set) > 16:
        tmp = random.sample(keyframe_id_set, 16)
        keyframe_id_set = set(tmp)
    elif len(keyframe_id_set) < 16:
        tmp = list(keyframe_id_set)
        tmp.sort()
        maxframe = tmp[len(tmp) - 1]
        while len(keyframe_id_set) < 16:
            newframe = random.randint(0, maxframe)
            keyframe_id_set.add(newframe)

    # save all keyframes as image
    cap = cv2.VideoCapture(str(videopath))
    curr_frame = None
    keyframes = []
    success, frame = cap.read()
    idx = 0
    num = 0
    while (success):
        if idx in keyframe_id_set:
            # name = str(id) + "_keyframe_" + str(idx) + ".jpg"
            name = str(id) + "_keyframe_" + str(num) + ".jpg"
            rframe = cv2.resize(frame, (112, 112))  # 压缩为112*112
            cv2.imwrite(dir + name, rframe)
            keyframe_id_set.remove(idx)
            num += 1
        idx = idx + 1
        success, frame = cap.read()
    cap.release()
    print(id, "over!")


if __name__ == '__main__':
    for i in movieID:
        kf_do(i)
    print("all success!")
