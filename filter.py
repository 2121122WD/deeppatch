"""新增代码：在数据集上运行网络区别干净样本和干扰样本"""
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from vgg import VGG
from densenet import DenseNet121
from resnet import ResNet101
from mobilenetv2 import MobileNetV2
from loader import GN
import numpy as np
import pathlib


def filter_samples(noise_name, net_name):
    output_path = pathlib.Path(f"{noise_name}{net_name}T11_index.npy")
    if output_path.is_file():
        print(f"Filtered data already exists for {noise_name} and {net_name}.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据集
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = GN(data_path=f'data/{noise_name}_train.npy', label_path='data/labels_train.npy', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    # 初始网络
    model_map = {
        'vgg16': VGG('VGG16'),
        'densenet121': DenseNet121(),
        'resnet101': ResNet101(),
        'mobilenetv2': MobileNetV2(),
    }
    if net_name not in model_map:
        raise ValueError(f"Model {net_name} is not supported.")

    net = model_map[net_name].to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # 加载预训练权重
    weight_path = f'net_weight/{net_name}_ckpt.pth'
    weight = torch.load(weight_path)
    net.load_state_dict(weight['net'])

    # 实现了对输入样本的分类（判断干净样本或噪声样本）
    T11_index = [] # 保存预测错误样本的索引
    T12_index = [] # 保存预测正确样本的索引

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1) # 获取预测结果
            if predicted != targets:
                T11_index.append(batch_idx) # 如果预测错误，加入 T11
            else:
                T12_index.append(batch_idx) # 如果预测错误，加入 T12

    m = np.array(T11_index)
    np.save(noise_name+net_name+'T11_index.npy', m)
    n = np.array(T12_index)
    np.save(noise_name+net_name+'T12_index.npy', n)
