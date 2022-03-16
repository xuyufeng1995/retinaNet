import os
import shutil
from lxml import etree
import pandas as pd
import torch
import seaborn as sn
import matplotlib.pyplot as plt


def parse_xml_to_dict(xml):
    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


# for anno in os.listdir("/home/xuyufeng/dataset/Loiter/VOCdevkit/VOC2022/Annotations"):
#     with open(os.path.join("/home/xuyufeng/dataset/Loiter/VOCdevkit/VOC2022/Annotations", anno)) as fid:
#         xml_str = fid.read()
#     xml = etree.fromstring(xml_str)
#     data = parse_xml_to_dict(xml)["annotation"]
#     if data["filename"] == "img1_gt.avi":
#         print(anno)

# 1、数据集大小显示
count = 0
for image in os.listdir("/home/xuyufeng/dataset/Loiter/VOCdevkit/VOC2022/JPEGImages"):
    if os.path.exists(os.path.join("/home/xuyufeng/dataset/Loiter/VOCdevkit/VOC2022/Annotations", image.replace("jpg", "xml"))):
        count += 1

with open("/home/xuyufeng/dataset/Loiter/VOCdevkit/VOC2022/ImageSets/Main/train.txt", "r") as f:
    train_list = f.read().strip().split("\n")

with open("/home/xuyufeng/dataset/Loiter/VOCdevkit/VOC2022/ImageSets/Main/val.txt", "r") as f:
    val_list = f.read().strip().split("\n")
print('Number of image are: {}, train are: {}, val are: {}'.format(count, len(train_list), len(val_list)))
# Number of image are: 64507, train are: 58056, val are: 6451
# 2、框大小显示

boxes = []
labels = []
small, medium, large = 0, 0, 0
for anno in os.listdir("/home/xuyufeng/dataset/Loiter/VOCdevkit/VOC2022/Annotations"):
    with open(os.path.join("/home/xuyufeng/dataset/Loiter/VOCdevkit/VOC2022/Annotations", anno)) as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = parse_xml_to_dict(xml)["annotation"]
    iw, ih = int(data["size"]["width"]), int(data["size"]["height"])
    for obj in data["object"]:
        w = (float(obj["bndbox"]["xmax"]) - float(obj["bndbox"]["xmin"])) / iw
        h = (float(obj["bndbox"]["ymax"]) - float(obj["bndbox"]["ymin"])) / ih
        x = (float(obj["bndbox"]["xmin"]) + w / 2) / iw
        y = (float(obj["bndbox"]["ymin"]) + h / 2) / ih
        if obj["name"] == "person" and w*h*iw*ih > 1:
            boxes.append([x, y, w, h])
            labels.append(0)
            if w*h*iw*ih < 32*32:
                small += 1
            elif w*h*iw*ih < 96*96:
                medium += 1
            else:
                large += 1

print(small, medium, large)

# area = w * h
x = pd.DataFrame(boxes, columns=["x", "y", "width", "height"])
print(x)
sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
plt.savefig('labels_correlogram.jpg', dpi=200)
plt.close()
# 280367 * 4个目标框
