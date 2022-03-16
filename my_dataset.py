from abc import ABC

from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree
import random
import cv2
import math
import numpy as np


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets["boxes"])

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets["boxes"][:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets["boxes"] = torch.as_tensor(new_bboxes, dtype=targets["boxes"].dtype)

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), targets


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


class VOCDataSet(Dataset):
    """读取解析PASCAL VOC2007/2012数据集"""

    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt",
                 mosaic_prob=1.0, mixup_prob=1.0, img_size=(1088, 640)):
        assert year in ["2022", "2012"], "year must be in ['2022', '2012']"
        self.root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        # read train.txt or val.txt file
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines() if len(line.strip()) > 0]

        # check file
        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), "not found '{}' file.".format(xml_path)

        # read class_indict
        json_file = './pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        json_file = open(json_file, 'r')
        self.class_dict = json.load(json_file)
        json_file.close()

        self.transforms = transforms

        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.img_size = img_size

    def __len__(self):
        return len(self.xml_list)

    def pull_item(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, xml_path.strip().split("/")[-1].replace("xml", "jpg"))
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        assert "object" in data, "{} lack of object information.".format(xml_path)

        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            if xmax <= xmin or ymax <= ymin or obj["name"] != "person":
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target

    def __getitem__(self, idx):
        image, target = self.masoic(idx)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def masoic(self, idx):
        if random.random() < self.mosaic_prob:
            mosaic_labels = {"boxes": torch.Tensor(),
                             "image_id": torch.tensor([idx]),
                             "labels": torch.Tensor(),
                             "iscrowd": torch.Tensor()}

            input_w, input_h = self.img_size[0], self.img_size[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self.xml_list) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, target = self.pull_item(index)
                w0, h0 = img.size  # orig wh

                scale = min(1. * input_w / w0, 1. * input_h / h0)
                img = img.resize((int(w0 * scale), int(h0 * scale)), Image.BILINEAR)

                # generate output mosaic image
                w, h = img.size
                if i_mosaic == 0:
                    mosaic_img = Image.new('RGB', (input_w * 2, input_h * 2), (114, 114, 114))

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img.paste(img.crop((s_x1, s_y1, s_x2, s_y2)), (l_x1, l_y1, l_x2, l_y2))
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                mosaic_labels["boxes"] = torch.cat([mosaic_labels["boxes"],
                                                    target["boxes"] * scale + torch.tensor(
                                                        [padw, padh, padw, padh]).reshape(1, -1).repeat(
                                                        target["boxes"].shape[0], 1)], 0)
                mosaic_labels["labels"] = torch.cat([mosaic_labels["labels"], target["labels"]], 0)
                mosaic_labels["iscrowd"] = torch.cat([mosaic_labels["iscrowd"], target["iscrowd"]], 0)

            if mosaic_labels["boxes"].size(0) > 0:
                mosaic_labels["boxes"][:, 0] = torch.clamp(mosaic_labels["boxes"][:, 0], 0, 2 * input_w)
                mosaic_labels["boxes"][:, 1] = torch.clamp(mosaic_labels["boxes"][:, 1], 0, 2 * input_h)
                mosaic_labels["boxes"][:, 2] = torch.clamp(mosaic_labels["boxes"][:, 2], 0, 2 * input_w)
                mosaic_labels["boxes"][:, 3] = torch.clamp(mosaic_labels["boxes"][:, 3], 0, 2 * input_h)

            mosaic_img, mosaic_labels = random_affine(
                cv2.cvtColor(np.asarray(mosaic_img), cv2.COLOR_RGB2BGR),
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=10.0,
                translate=0.1,
                scales=(0.5, 1.5),
                shear=2.0,
            )

            # if random.random() < self.mixup_prob and len(mosaic_labels):
            #     mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels)

            # remove area = 0 box
            mosaic_labels["area"] = (mosaic_labels["boxes"][:, 2] - mosaic_labels["boxes"][:, 0]) * (
                        mosaic_labels["boxes"][:, 3] - mosaic_labels["boxes"][:, 1])
            mark = mosaic_labels["area"] > 1  # 面积大于1，防止目标框小于0的情况
            mosaic_labels["boxes"] = mosaic_labels["boxes"][mark]
            mosaic_labels["area"] = mosaic_labels["area"][mark]
            mosaic_labels["labels"] = mosaic_labels["labels"][mark]
            mosaic_labels["iscrowd"] = mosaic_labels["iscrowd"][mark]

            # float 2 int64
            mosaic_labels["labels"] = mosaic_labels["labels"].long()
            mosaic_labels["iscrowd"] = mosaic_labels["iscrowd"].long()

            image = mosaic_img
            target = mosaic_labels

        else:
            image, target = self.pull_item(idx)

        return image, target

    def mixup(self, origin_img, origin_labels, mixup_scale=(0.5, 1.5)):
        jit_factor = random.uniform(*mixup_scale)
        cp_labels = {"boxes": torch.Tensor()}
        while cp_labels["boxes"].size(0) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            img, cp_labels = self.pull_item(cp_index)

        cp_img = Image.new('RGB', (self.img_size[0], self.img_size[1]), (114, 114, 114))

        cp_scale_ratio = min(self.img_size[0] / img.width, self.img_size[1] / img.height)
        resized_img = img.resize((int(img.width * cp_scale_ratio), int(img.height * cp_scale_ratio)), Image.BILINEAR)

        cp_img.paste(resized_img, (0, 0, resized_img.width, resized_img.height))
        cp_img = cp_img.resize((int(cp_img.width * jit_factor), int(cp_img.height * jit_factor)), Image.BILINEAR)
        cp_scale_ratio *= jit_factor

        origin_w, origin_h = cp_img.size
        target_w, target_h = origin_img.size
        padded_img = Image.new('RGB', (max(origin_w, target_w), max(origin_h, target_h)), (0, 0, 0))
        padded_img.paste(cp_img, (0, 0, origin_w, origin_h))

        x_offset, y_offset = 0, 0
        if padded_img.width > target_w:
            x_offset = random.randint(0, padded_img.width - target_w - 1)
        if padded_img.height > target_h:
            y_offset = random.randint(0, padded_img.height - target_h - 1)
        padded_cropped_img = Image.new('RGB', (target_w, target_h), (0, 0, 0))
        padded_cropped_img.paste(padded_img.crop((x_offset, y_offset, x_offset + target_w, y_offset + target_h)), (0, 0, target_w, target_h))

        cp_labels["boxes"][:, 0::2] = torch.clamp(cp_labels["boxes"][:, 0::2] * cp_scale_ratio, 0, origin_w)
        cp_labels["boxes"][:, 1::2] = torch.clamp(cp_labels["boxes"][:, 1::2] * cp_scale_ratio, 0, origin_h)

        cp_labels["boxes"][:, 0::2] = torch.clamp(cp_labels["boxes"][:, 0::2] - x_offset, 0, target_w)
        cp_labels["boxes"][:, 1::2] = torch.clamp(cp_labels["boxes"][:, 1::2] - y_offset, 0, target_h)

        origin_labels["boxes"] = torch.cat([origin_labels["boxes"], cp_labels["boxes"]], dim=0)
        origin_labels["labels"] = torch.cat([origin_labels["labels"], cp_labels["labels"]], dim=0)
        origin_labels["iscrowd"] = torch.cat([origin_labels["iscrowd"], cp_labels["iscrowd"]], dim=0)

        origin_img = Image.blend(origin_img, padded_cropped_img, alpha=0.5)

        return origin_img, origin_labels

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        image, target = self.pull_item(idx)
        data_width, data_height = image.size

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class YOLODataSet(Dataset):
    """读取解析PASCAL VOC2007/2012数据集"""

    def __init__(self, base_root, transforms=None, txt_name: str = "tr_list.txt"):
        self.root = base_root
        self.img_root = os.path.join(self.root, "images")
        self.labels_root = os.path.join(self.root, "labels")

        # read train.txt or val.txt file
        txt_path = os.path.join(self.root, "metadata", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:
            self.img_list = [_.strip() for _ in read.readlines() if len(_.strip()) > 0]

        # check file
        assert len(self.img_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        for img_path in self.img_list:
            assert os.path.exists(img_path), "not found '{}' file.".format(img_path)

        # read class_indict
        json_file = 'pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        json_file = open(json_file, 'r')
        self.class_dict = json.load(json_file)
        json_file.close()

        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # read label
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))
        iw, ih = image.size

        label_path = img_path.replace("images", "labels").replace("jpg", "txt")
        with open(label_path) as fid:
            data = fid.read().strip().split("\n")

        boxes = []
        labels = []
        for obj in data:
            bbox = obj.split(" ")
            xmin = (float(bbox[1]) - float(bbox[3]) / 2 ) * iw
            xmax = (float(bbox[1]) + float(bbox[3]) / 2 ) * iw
            ymin = (float(bbox[2]) - float(bbox[4]) / 2 ) * ih
            ymax = (float(bbox[2]) + float(bbox[4]) / 2 ) * ih
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(bbox[0]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))
        data_width, data_height = image.size
        return data_height, data_width

    def coco_index(self, idx):
        """
         该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
         由于不用去读取图片，可大幅缩减统计时间

         Args:
             idx: 输入需要获取图像的索引
        """
        image, target = self.__getitem__(idx)
        _, data_height, data_width = image.shape
        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


# import transforms
# from draw_box_utils import draw_box
# from PIL import Image
# import json
# import matplotlib.pyplot as plt
# import torchvision.transforms as ts
# import random
#
# # read class_indict
# category_index = {}
# try:
#     json_file = open('./pascal_voc_classes.json', 'r')
#     class_dict = json.load(json_file)
#     category_index = {v: k for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)
#
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
#
# # load train data set
# train_data_set = YOLODataSet("/home/xuyufeng/projects/python/yolov5/datasets/", data_transform["train"], "tr_list.txt")
# print(len(train_data_set))
# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target = train_data_set[index]
#     img = ts.ToPILImage()(img)
#     draw_box(img,
#              target["boxes"].numpy(),
#              target["labels"].numpy(),
#              [1 for i in range(len(target["labels"].numpy()))],
#              category_index,
#              thresh=0.5,
#              line_thickness=5)
#     plt.imshow(img)
#     plt.show()
