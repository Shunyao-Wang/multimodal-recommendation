from fishervector import FisherVectorGMM
# from common import util
import scipy.io.wavfile as wav
import numpy as np
import speechpy
import os, fnmatch

#file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test.wav')

list_of_files = os.listdir('../audio/')
pattern = '*.wav'
for entry in list_of_files:
    if fnmatch.fnmatch(entry, pattern):
       fs, signal = wav.read('audio/'+entry)
       signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.97)
       mfcc = speechpy.feature.mfcc(signal_preemphasized, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                     num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
       mfcc_cmvn = speechpy.processing.cmvnw(mfcc, win_size=301, variance_normalization=True)
       deriv = speechpy.processing.derivative_extraction(mfcc_cmvn, 1)
       deriv_2 = speechpy.processing.derivative_extraction(deriv, 1)
       mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)
       new = np.concatenate((mfcc_cmvn, deriv, deriv_2), axis=1)
       shape = [3687, 13, 0] #[3687, 13, 3] # e.g. SIFT image features
       image_data = mfcc_feature_cube
       fv_gmm = FisherVectorGMM(n_kernels=20).fit(image_data)
       #fv_gmm = FisherVectorGMM().fit_by_bic(image_data, choices_n_kernels=[2,5,10,20])
       image_data_test = image_data[:20] # use a fraction of the data to compute the fisher vectors
       fv = fv_gmm.predict(image_data_test)
       np.savetxt(entry+"_fishers.txt", fv_gmm)
        #print(fv)
