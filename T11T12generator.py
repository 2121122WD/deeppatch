"""过读取噪声样本（T11）和干净样本（T12）的索引，从训练数据中提取对应的样本及其标签"""
import numpy as np
import pathlib


def generator(noise_name, net_name):
    path = pathlib.Path(noise_name+net_name+'T11_labels.npy')
    if path.is_file():
        return

    T11_index = np.load(noise_name+net_name+'T11_index.npy')
    T12_index = np.load(noise_name+net_name+'T12_index.npy')

    data = np.load('data/'+noise_name+'_train.npy')
    T11 = data[T11_index]
    T12 = data[T12_index]
    np.save(noise_name+net_name+'T11.npy', T11)
    np.save(noise_name+net_name+'T12.npy', T12)

    labels = np.load('data/labels_train.npy')
    T11_labels = labels[T11_index]
    T12_labels = labels[T12_index]
    np.save(noise_name+net_name+'T11_labels.npy', T11_labels)
    np.save(noise_name+net_name+'T12_labels.npy', T12_labels)
