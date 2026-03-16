# 概况

## 用途

本仓库用于HYX进行机器学习计算

## 编译环境确认

conda后请运行test.py
其中 pyg-lib 为 PyTorch Geometric 的底层优化库,建议安装,非必选

## 数据集下载地址

!!!请将下载好的molecules.zip置于ZINC文件夹下,不必解压!!!

https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1

!!!请将下载好的molecules.zip置于ZINC文件夹下,不必解压!!!

## 储存路径

执行前请修改数据集地址 root_path

## 静/动态调节

默认为动态，若要对比静态性能请将
配置选项中
use_sinkhorn = True
改为False

## 数据调节

若计算能力不足请修改
sinkhorn_iters = 10         # Sinkhorn迭代次数
hidden_dim=64
num_layers=4
...
