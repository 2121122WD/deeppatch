"""将噪声样本（T11）和干净样本（T12）的特征合并为一个特征数据集，同时生成对应的标签"""
import numpy as np
import pathlib


def merge(noise_name, net_name):
    path = pathlib.Path(noise_name+net_name+'feature_for_C.npy')
    if path.is_file():
        return

    T11_feature = np.load(noise_name+net_name+'feature_T11.npy')
    num1 = T11_feature.shape[0]
    T12_feature = np.load(noise_name+net_name+'feature_T12.npy')
    num2 = T12_feature.shape[0]

    feature = np.concatenate((T11_feature, T12_feature))
    np.save(noise_name+net_name+'feature_for_C.npy', feature)

    labels = np.concatenate((np.array([0] * num1, dtype=np.int64), np.array([1] * num2, dtype=np.int64)))
    np.save(noise_name+net_name+'labels_for_C.npy', labels)
