from moviepy.editor import *
from python_speech_features import *
import scipy.io.wavfile as wav
import numpy as np

file = open('../data/movieID.txt', 'r')
# file = open('../data/testMovieID.txt', 'r')
movieID = [int(x) for x in file]
file.close()

# data_dir = "../data/"  # 视频音频额外保存的目录
data_dir = "F:/data/"


# 从视频中提取音频
def v2a(id):
    video_file = data_dir + "videos/" + id + "_video.mp4"
    video = VideoFileClip(video_file)
    audio = video.audio
    audio_file = data_dir + "audios/" + id + ".wav"
    audio.write_audiofile(audio_file)
    print(audio_file + " saved success!")


# 从wav中提取mfcc特征
def getMFCC(id):
    audio_file = data_dir + "audios/" + id + ".wav"
    fs, signal = wav.read(audio_file)
    wav_feature = mfcc(signal, fs)
    d_mfcc_feat = delta(wav_feature, 1)
    d_mfcc_feat2 = delta(wav_feature, 2)
    feature = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
    print(id + " MFCC shape :", feature.shape)
    return feature


if __name__ == '__main__':
    for mID in movieID:
        mID = str(mID)
        v2a(mID)
        mfcc_feature = getMFCC(mID)
        save_file = data_dir + "mfccs/" + mID + "_MFCC.npy"
        np.save(save_file, mfcc_feature)
        print(save_file + " saved!")
