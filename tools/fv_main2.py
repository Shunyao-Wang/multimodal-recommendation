# encoding=utf-8
# python2 版本运行
import fisher_vector2 as fv
import numpy as np

DATAPATH = "../data/"  # MFCC保存的目录
# DATAPATH = "F:/data/"
# DATAPATH = "/home/wang/dev/data/" # ubuntu

file = open(DATAPATH + 'movieID.txt', 'r')
movieID = [int(x) for x in file]
file.close()
print(movieID)

K = 20  # 高斯混合模型的聚类中心数
D = 39

result_list = np.zeros((len(movieID), (2 * D + 1) * K), dtype=np.float32)
tmp = 0
for id in movieID:
    mfcc_file = DATAPATH + "mfccs/" + str(id) + "_MFCC.npy"
    mfcc = np.load(mfcc_file)  # MFCC特征维数为D，即39
    means, covs, weights = fv.generate_gmm(DATAPATH, mfcc, K)
    # fv输出的向量维度为(2*D+1)*K
    try:
        result = fv.fisher_vector(mfcc, means, covs, weights)
        result_list[tmp] = result
        print("id=" + str(id) + " over")
    except ValueError:
        print("id=" + str(id) + " fail!!!!!!!!!!!!!!!!!!!!!!!")
    tmp += 1
savefile = DATAPATH + "features/audio_feature.npy"
np.save(savefile, result_list)
# print(result_list)
print(savefile + " saved! shape:", result_list.shape)
