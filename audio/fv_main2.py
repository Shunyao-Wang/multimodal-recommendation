#encoding=utf-8
import fisher_vector2 as fv
import numpy as np

# 训练集，需要预先准备，矩阵列数为D
train = np.random.rand(10000,39)

# generate_gmm第三个参数为聚类中心K
means, covs, weights = fv.generate_gmm("../data",train,3)

# fv输出的向量维度为(2*D+1)*K
test = np.random.rand(666,39)
result = fv.fisher_vector(train,means,covs,weights)
print(result)
print(len(result))
