import video.c3d_model as c3d
import numpy as np
import cv2

DATAPATH = "../data/"
# DATAPATH = "F:/data/"
# DATAPATH = "/home/wang/dev/data/" # ubuntu

WEIGHTS_PATH = 'F:/data/sports1M_weights_tf.h5'
file = open(DATAPATH + 'movieID.txt', 'r')
movieID = [int(x) for x in file]
file.close()
print(movieID)

if __name__ == '__main__':
    # 获取加载权重模型
    model = c3d.get_model(with_weights=WEIGHTS_PATH)
    # 获取fc7层输出模型
    model = c3d.Model(inputs=model.input, outputs=model.get_layer('fc7').output)
    # predict进行预测，输入向量[n, 16, 112, 112, 3], 输出向量[n, 4096]

    train = np.zeros((len(movieID), 16, 112, 112, 3), dtype=np.uint8)
    for i in range(len(movieID)):
        mid = movieID[i]
        path = DATAPATH + "extract_result/" + str(mid) + "_keyframe_"
        for j in range(16):
            tmp = cv2.imread(path + str(j) + ".jpg")
            train[i][j] = tmp
        print(path, "over!")
    result = model.predict(train)
    savefile = DATAPATH + "features/video_feature.npy"
    np.save(savefile, result)
    print(train.shape, result.shape, "saved success!")
