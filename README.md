# NeuroPatch
Implementations for networks (VGG16, DenseNet121, ResNet101 and MobileNetV2) for CIFAR-10 come from [here](https://github.com/kuangliu/pytorch-cifar).

主要新增代码讲解：
## filter.py
filter.py 脚本的主要功能是根据指定深度学习模型的预测结果，将输入样本分为两类：
- 噪声样本 (T11)：
模型预测错误的样本（即预测标签与真实标签不匹配的样本）。
- 干净样本 (T12)：
模型预测正确的样本（即预测标签与真实标签匹配的样本）。
该脚本会对输入数据集进行处理，分类出噪声和干净样本，并将对应的索引保存为 .npy 文件，供后续使用。

filter.py 中的 filter_samples 函数主要包括以下步骤：

```加载数据集：
从 .npy 文件读取训练数据集和对应标签。
对数据应用标准的预处理操作（如归一化）。
```

```初始化模型：
加载预训练的深度学习模型（支持 VGG16、DenseNet121、ResNet101 和 MobileNetV2）。
加载对应的模型权重。
```

```分类样本：
遍历数据集，对每个样本进行评估。
根据预测标签与真实标签的对比，将样本分类为噪声样本（预测错误）或干净样本（预测正确）。
```

```保存结果：
将噪声样本 (T11) 和干净样本 (T12) 的索引保存为 .npy 文件：
<noise_name><net_name>T11_index.npy
<noise_name><net_name>T12_index.npy
```
## T11T12generator.py
通过读取噪声样本（T11）和干净样本（T12）的索引，从训练数据中提取对应的样本及其标签，并将其分别保存为 .npy 文件

## Environment:
```
python==3.6.12
pytorch==1.9.0
scikit-learn==0.24.2
```
