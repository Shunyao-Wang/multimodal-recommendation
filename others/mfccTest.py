from moviepy.editor import *

# video = VideoFileClip('../8586_video.mp4')
# audio = video.audio
# audio.write_audiofile('test.wav')
import matplotlib.pyplot as plt
from python_speech_features import mfcc as pmfcc
import scipy.io.wavfile as wav
filepath = "test.wav"
(rate,sig) = wav.read(filepath)
amfcc = pmfcc( sig, rate ).T
