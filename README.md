# 数字图像处理 2021 秋 课程设计 (DIP 2021 Fall Final Project)

### 1. 安装相关依赖

本项目所需依赖已经列在 requirement.txt 中，运行以下代码安装依赖：

```shell
$ pip install -r requirement.txt
```

### 2. 生成数据集

车票训练集图片在 training_data 目录下，注释文件名为 training_annotation.txt, 在当前目录下运行以下代码：

```Shell
$ python segment.py --image-dir training_data --annotation-file training_annotation.txt
```

则数据集将被分别存入

- number_data/train
- number_data/val
- letter_data/train
- letter_data/val

若需要扩充数据集，提供的额外车票图片在 extra_data 目录下，在当前目录运行 extra.ipynb 文件，则扩充数据集被自动存入以上目录。

### 3. 训练模型

在当前目录下直接运行以下代码：

```Shell
$ python train.py 
```

训练好的模型将被存入 models 目录中。

### 4. 预测输出

车票测试集图片在 test_data 目录下，注释文件名为 annotation.txt, 在当前目录下运行以下代码：

```Shell
$ python predict.py --input-dir test_data --annotation-file annotation.txt --output-dir segments --prediction-file prediction.txt
```

则定位并分割好的票面被存入 segments 目录中，预测输出的21位码与7位码结果被写入 prediction.txt 文件中。