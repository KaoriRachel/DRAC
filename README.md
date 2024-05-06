# DRAC
本项目为DRAC2022竞赛的任务二（高血压病变图像质量评估）的项目代码
## 安装
你可以通过以下步骤安装代码
1. 克隆仓库 git clone https://github.com/KaoriRachel/DRAC.git
2. 安装依赖项
3. 下载图片所用数据集以及标签（标签已在仓库中） https://jbox.sjtu.edu.cn/l/P1ZjOw
4. 下载需要的模型，模型后缀.h5为TensorFlow模型，使用Train.py文件训练和生成测试结果；后缀.pth为torch模型，使用Train_mix.py文件训练，Test_mix.py文件生成测试结果
## 运行
###
```bash
python Train.py
```
Train.py使用TensorFlow实现。这条命令将会调用best_model训练且生成测试结果

###
```bash
python Train_mix.py
```
Train_mix.py使用PyTorch实现。旨在使用mix_up方法训练模型。
```bash
python Test_mix.py
```
Test_mix.py与Train_mix.py相搭配，调用其训练的模型生成测试结果

如要使用PyTorch版本，在调用不同模型时需要调整模型最后一层的网络结构。
