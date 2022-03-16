import os
import time
import json

import torch
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import RetinaNet
from backbone import resnet50_fpn_backbone, LastLevelP6P7
from draw_box_utils import draw_box
import cv2
import numpy as np


def create_model(num_classes):
    # resNet50+fpn+retinanet
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     returned_layers=[2, 3, 4],
                                     extra_blocks=LastLevelP6P7(256, 256))
    model = RetinaNet(backbone, num_classes)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    # 注意：不包含背景
    model = create_model(num_classes=1)

    # load train weights
    train_weights = "./runs/exp_multi_gpu_ap0.8701/weights/best.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])

    # init
    model.eval()  # 进入验证模式
    img_height, img_width = 1080, 1920
    init_img = torch.zeros((1, 3, img_height, img_width), device=device)
    model(init_img)

    # save video
    video = cv2.VideoWriter("VideoTest.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps=24, frameSize=(1920, 1080))

    # load videos
    cap = cv2.VideoCapture("./videos/canting1fdonglouti_220221.mp4")
    while True:
        success, frame = cap.read()
        if not success:
            break

        original_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            # t_end = time_synchronized()
            # print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            draw_box(original_img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=0.5,
                     line_thickness=3)

            img = cv2.cvtColor(np.asarray(original_img), cv2.COLOR_RGB2BGR)
            video.write(img)

    video.release()
    cap.release()


if __name__ == '__main__':
    main()

