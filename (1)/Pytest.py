from tensorflow.contrib.keras.api.keras.models import load_model
from cifar_data_load import GetTestDataByLabel
import numpy as np

if __name__ == '__main__':
    # 载入训练好的模型
    model = load_model("lenet-no-activation-model.h5")
    # 获取测试集的数据
    X = GetTestDataByLabel('data')
    Y = GetTestDataByLabel('labels')
    # 统计预测正确的图片的数目
    print(np.sum(np.equal(Y, np.argmax(model.predict(X), 1))))
