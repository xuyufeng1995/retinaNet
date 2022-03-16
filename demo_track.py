import argparse
import os
import time
import datetime
import cv2
import torch
from backbone import resnet50_fpn_backbone, LastLevelP6P7
from network_files import RetinaNet
from tracker.byte_tracker import BYTETracker
from tracker.timer import Timer
import json
from torchvision import transforms
import numpy as np


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument(
        "--path", default="./videos/门卫室内_20210713100000-20210713101000_1.mp4", help="path to images or video"
    )
    parser.add_argument("--num_classes", default=1, type=int, help="classes number")
    parser.add_argument(
        "--output_dir",
        default="./runs",
        type=str,
        help="whether to save the inference result of image/video",
    )

    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.5, type=float, help="test conf")
    parser.add_argument("--nms", default=0.5, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=(640, 1088), type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    return parser


def create_model(num_classes):
    # resNet50+fpn+retinanet
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     returned_layers=[2, 3, 4],
                                     extra_blocks=LastLevelP6P7(256, 256))
    model = RetinaNet(backbone, num_classes)

    return model


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        device=torch.device("cpu"),
        fp16=False,
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.conf
        self.nmsthre = exp.nms
        self.test_size = exp.tsize
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        img = self.transforms(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0).float().to(self.device)

        # img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            predictions = self.model(img)[0]

            predict_boxes = predictions["boxes"].to("cpu")
            predict_classes = predictions["labels"].unsqueeze(1).to("cpu")
            predict_scores = predictions["scores"].unsqueeze(1).to("cpu")

            predictions = torch.cat([predict_boxes, predict_scores], dim=1)

            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))

        return predictions


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im


def imageflow_demo(predictor, output_dir, current_time, args):
    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = os.path.join(output_dir, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    print(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            print('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        success, frame = cap.read()
        if frame_id % 5 != 0:
            frame_id += 1
            continue

        if success:
            outputs = predictor.inference(frame, timer)
            # for box in outputs:
            #     if box[4].item() < args.track_thresh:
            #         continue
            #     x1 = box[0].int().item()
            #     y1 = box[1].int().item()
            #     x2 = box[2].int().item()
            #     y2 = box[3].int().item()
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            # cv2.imwrite("test.jpg", frame)

            if outputs[0] is not None:
                online_targets = tracker.update(outputs, args.tsize)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    frame, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = frame
            vid_writer.write(online_im)
            # cv2.waitKey(1)
        else:
            break
        frame_id += 1

    # res_file = os.path.join(output_dir, f"{timestamp}.txt")
    # with open(res_file, 'w') as f:
    #     f.writelines(results)
    print("save success!!!")


def main(args):
    print("Args: {}".format(args))

    vis_folder = os.path.join(args.output_dir, "track")
    os.makedirs(vis_folder, exist_ok=True)

    device = torch.device("cuda" if args.device == "gpu" else "cpu")

    model = create_model(num_classes=args.num_classes)

    # load train weights
    assert os.path.exists(args.ckpt), "{} file dose not exist.".format(args.ckpt)
    print("loading checkpoint")
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.to(device)
    print("loaded checkpoint done.")

    if args.fp16:
        model = model.half()  # to FP16

    model.eval()  # 进入验证模式

    predictor = Predictor(model, args, device, args.fp16)
    current_time = time.localtime()
    if args.demo == "video":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
