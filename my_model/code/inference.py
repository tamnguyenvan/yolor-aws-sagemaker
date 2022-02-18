import os
import io
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import argparse
import time
import math
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from PIL import Image


names = ['person', 'hardhat', 'glasses', 'vest', 'gloves', 'hand']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
augment = False
conf_thres = 0.25
iou_thres = 0.45
classes = None
agnostic_nms = False


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep, :]


def slice_images_wo_labels(image, tile_size, overlap_ratio=0.25):
    height, width = image.shape[:2]
    overlap_w, overlap_h = int(overlap_ratio * tile_size), int(overlap_ratio * tile_size)

    if width < tile_size:
        nx = 1
    else:
        nx = math.ceil((width - overlap_w) / (tile_size - overlap_w))

    if height < tile_size:
        ny = 1
    else:
        ny = math.ceil((height - overlap_h) / (tile_size - overlap_h))

    tiles = []
    for i in range(1, ny+1):
        for j in range(1, nx+1):
            x1 = (j - 1) * (tile_size - overlap_w)
            x2 = x1 + tile_size
            if x2 > width:
                x2 = width
                x1 = max(0, x2 - tile_size)

            y1 = (i - 1) * (tile_size - overlap_h)
            y2 = y1 + tile_size
            if y2 > height:
                y2 = height
                y1 = max(0, y2 - tile_size)

            tile_image = image[y1:y2, x1:x2, :].copy()
            tiles.append((x1, y1, tile_image))
    return tiles


def detect(model, im0, imgsz, device):
    img = letterbox(im0, new_shape=imgsz, auto_size=64)[0]

    # Convert
    img = img[:, :, :].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t2 = time_synchronized()

    # Process detections
    dets = []
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                xyxy = [x.item() for x in xyxy]
                conf = conf.item()
                cls = cls.item()
                dets.append([xyxy, conf, cls])
    
    return dets


def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'my_model', 'model.pth')
    model = attempt_load(model_path, map_location=device)
    return model


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/octet-stream':
        return Image.open(io.BytesIO(request_body)).convert('RGB')
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        raise Exception(f'Unsupported content type: {request_content_type}')


def predict_fn(input_data, model):
    # model.eval()
    # model.to(device)

    # with torch.no_grad():
    #     if isinstance(input_data, torch.Tensor):
    #         input_data = input_data.permute((2, 0, 1)).contiguous()
    #         input_data = input_data.float()
    #         defaults = transforms.Compose([
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])]
    #         )
    #     else:
    #         defaults = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])]
    #         )
    #     images = defaults(input_data)
    #     images = images.to(device)

    #     preds = model([images])
    #     # Send predictions to CPU if not already
    #     preds = [{k: v.to(torch.device('cpu')) for k, v in p.items()} for p in preds]
    
    # results = []
    # for pred in preds:
    #     # Convert predicted ints into their corresponding string labels
    #     result = ([classes[val] for val in pred['labels']], pred['boxes'].numpy().tolist(), pred['scores'].numpy().tolist())
    #     results.append(result)

    model.eval()
    imgsz = check_img_size(imgsz, s=model.stride.max())
    with torch.no_grad():
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.detach().cpu().numpy()
    image = input_data.copy()

    overlap_ratio = 0.25
    tiles = slice_images_wo_labels(image, imgsz, overlap_ratio=overlap_ratio)

    preds = []
    height, width = input_data.shape[:2]
    for offset_x, offset_y, img in tiles:
        dets = detect(model, img, imgsz, device)
        if len(dets):
            for xyxy, conf, cls in dets:
                x1, y1, x2, y2 = xyxy
                x1 += offset_x
                y1 += offset_y
                x2 += offset_x
                y2 += offset_y

                preds.append([x1, y1, x2, y2, conf, cls])
    preds = np.array(preds)
    clss = np.unique(preds[:, 5])
    nms_preds = []
    for class_id in clss:
        pred = nms(preds[preds[:, 5] == class_id], iou_thres)
        nms_preds.append(pred)
    
    preds = np.concatenate(nms_preds, 0)

    out_data = preds[:, :4] / np.array([height, height, width, height])
    out_class = preds[:, 5:6]
    out_centers = (out_data[:, :2] + out_data[:, 2:4]) / 2
    out_wh = out_data[:, 2:4] - out_data[:, :2]
    out_data = np.concatenate([out_class, out_centers, out_wh], axis=1)
    # with open(txt_path, 'wt') as f:
    #     for row in out_data:
    #         class_id, cx, cy, w, h = row
    #         f.write('%d,%.4f,%.4f,%.4f,%.4f\n' % (class_id, cx, cy, w, h))
    # print('Saved output text as:', txt_path)

    results = []
    for class_id, cx, cy, w, h in out_data:
        results.append([class_id, names[class_id], cx, cy, w, h, width, height])
    return results