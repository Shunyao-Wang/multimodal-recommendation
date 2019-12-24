from others import fvtest2 as fv
import numpy as np

train = np.random.rand(10000,39)
train

means, covs, weights = fv.generate_gmm("./data",train,3)

test = np.random.rand(666,39)
result = fv.fisher_vector(train,means,covs,weights)
print(result)
print(len(result))