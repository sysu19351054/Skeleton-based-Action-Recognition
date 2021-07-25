# Skeleton-based-Action-Recognition
深度学习实验期末大作业

任务：
人体姿态序列分类

说明：
尝试了四种网络结构：
ResNet18（测试集准确率过低舍弃）
ResNet3D（最高测试集准确率 0.7948717948717948）
GRU（最高测试集准确率 0.7777777777777778）
LSTM（最高测试集准确率 0.8376068376068376）

文件结构：
将代码，mdl文件和data放在同一文件夹下

测试：
运行test.py，可得已经训练好的模型（包括GRU_best.mdl和LSTM_best.mdl）的测试集准确率
（ResNet3D的mdl文件过大，无法上传）

模型训练：
运行LSTM_train.py, ResNet3D_train.py, GRU_train.py 分别训练三个网络
会输出新训练模型的测试集准确率
新模型保存在LSTM.mdl, ResNet.mdl, GRU.mdl
