import os
import io
import time
import math
import json

import cv2
import torch
import torch.nn as nn
import numpy as np

from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords

from PIL import Image


names = ['person', 'hardhat', 'glasses', 'vest', 'gloves', 'hand']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
augment = False
conf_thres = 0.25
iou_thres = 0.45
classes = None
agnostic_nms = False


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros(0, 6)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


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
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

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


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    for w in weights if isinstance(weights, list) else [weights]:
        model = torch.load(w, map_location=map_location)['model'].float().fuse().eval()  # load FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    return model


def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'my_model', 'model.pth')
    model = attempt_load(model_path, map_location=device)
    return model


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    # if request_content_type == 'application/octet-stream':
    #     return Image.open(io.BytesIO(request_body)).convert('RGB')
    # else:
    #     # Handle other content-types here or raise an Exception
    #     # if the content type is not supported.
    #     raise Exception(f'Unsupported content type: {request_content_type}')
    assert request_content_type == "application/json"
    data = json.loads(request_body)["inputs"]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    return data



def predict_fn(input_data, model):
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