
## 文件下载
训练所需的ssd_weights.pth和主干的权值可以在百度云下载。  
链接: https://pan.baidu.com/s/1iUVE50oLkzqhtZbUL9el9w     
提取码: jgn8     

VOC数据集下载地址如下，里面已经包括了训练集、测试集、验证集（与测试集一样），无需再次划分：  
链接: https://pan.baidu.com/s/1-1Ej6dayrx3g0iAA88uY5A    
提取码: ph32   

## 训练步骤
### a、训练VOC07+12数据集
1. 数据集的准备   
本文使用VOC格式进行训练，训练前需要下载好VOC07+12的数据集，解压后放在根目录

2. 数据集的处理   
运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。   

3. 开始网络训练   
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。   


## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载，放入model_data，运行predict.py

## 评估步骤 
1. 本文使用VOC格式进行评估。VOC07+12已经划分好了测试集，无需利用voc_annotation.py生成ImageSets文件夹下的txt。
2. 运行get_map.py前修改map_mode=1/2/3，评估保存在map_out。


## Reference
https://github.com/bubbliiiing/ssd-pytorch
