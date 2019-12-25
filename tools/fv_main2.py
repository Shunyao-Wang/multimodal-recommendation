# encoding=utf-8
# python2 版本运行
import fisher_vector2 as fv
import numpy as np

file = open('../data/movieID.txt', 'r')
# file = open('../data/testMovieID.txt', 'r')
movieID = [int(x) for x in file]
file.close()
print(movieID)

# data_dir = "../data/"  # MFCC保存的目录
data_dir = "F:/data/"
# data_dir = "/home/wang/dev/data/" # ubuntu
K = 3  # 高斯混合模型的聚类中心数
D = 39

result_list = np.zeros((len(movieID), (2 * D + 1) * K), dtype=np.float32)
tmp = 0
for id in movieID:
    mfcc_file = data_dir + "mfccs/" + str(id) + "_MFCC.npy"
    mfcc = np.load(mfcc_file)  # MFCC特征维数为D，即39
    means, covs, weights = fv.generate_gmm(data_dir, mfcc, 3)
    # fv输出的向量维度为(2*D+1)*K
    result = fv.fisher_vector(mfcc, means, covs, weights)
    result_list[tmp] = result
    print("id=" + str(id) + " over")
    tmp += 1
savefile = "../data/features/audio_feature.npy"
np.save(savefile, result_list)
# print(result_list)
print(savefile + " saved! shape:", result_list.shape)
