# RetinaNet

## 该项目主要是来自pytorch官方torchvision模块中的源码
* https://github.com/pytorch/vision/tree/master/torchvision/models/detection


## 预训练权重下载地址（下载后放入backbone文件夹中）：
* ResNet50+FPN backbone: https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth
* 注意，下载的预训练权重记得要重命名，比如在train.py中读取的是```retinanet_resnet50_fpn_coco.pth```文件，
  不是```retinanet_resnet50_fpn_coco-eeacb38b.pth```
  
## 训练方法
* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 若要单GPU训练，直接使用train.py训练脚本
* 若要使用多GPU训练，使用```python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py```指令,```nproc_per_node```参数为使用GPU数量
* 如果想指定使用哪些GPU设备可在指令前加上```CUDA_VISIBLE_DEVICES=0,3```(例如我只要使用设备中的第1块和第4块GPU设备)
* ```CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py```

## 注意事项
* 在使用训练脚本时，注意要将'--data-path'(VOC_root)设置为自己存放'VOCdevkit'文件夹所在的**根目录**
* 由于带有FPN结构的Faster RCNN很吃显存，如果GPU的显存不够(如果batch_size小于8的话)建议在create_model函数中使用默认的norm_layer，
  即不传递norm_layer变量，默认去使用FrozenBatchNorm2d(即不会去更新参数的bn层),使用中发现效果也很好。
* 在使用预测脚本时，要将'train_weights'设置为你自己生成的权重路径。
* 使用validation文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时只需要修改'--num-classes'、'--data-path'和'--weights'即可，其他代码尽量不要改动


## 改进点
* 将回归损失函数l1_loss修改为smooth_l1_loss
* 添加mosaic和mixup
* 添加byteTrack跟踪代码
* 添加tensorboard显示代码
