import json
import yaml
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn

from models.common import Conv
from utils.metrics import ap_per_class
from utils.dataset import create_dataloader
from utils.torch_utils import time_synchronized
from utils.general import non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, box_iou


def inference(data,
              weights=None,
              batch_size=32,
              imgsz=640,
              conf_thres=0.001,
              iou_thres=0.65,  # for NMS
              augment=False,
              save_dir=Path(''),  # for saving images
              save_hybrid=False,  # for hybrid auto-labelling
              half_precision=True):
    cuda = not (opt.device.lower() == 'cpu') and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    # load model
    print('Loading pretrained model...')
    ckpt = torch.load(opt.weights, map_location=device)
    model = ckpt.get('ema').float().fuse().eval()
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    half = device.type != 'cpu' and half_precision
    if half:
        model.half()  # half precision only supported on CUDA

    model.eval()
    with open(data) as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)
    nc = int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    print()
    dataloader = create_dataloader(data['val'], imgsz, batch_size, gs, opt, pad=0.5, rect=True)[0]

    seen = 0
    s = ('%12s' * 6 + '%20s') % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map = 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap = [], [], []

    print()
    for batch_i, (img, aop, dop, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        aop = aop.to(device, non_blocking=True)
        aop = aop.half() if half else aop.float()  # uint8 to fp16/32
        aop /= 255.0  # 0 - 255 to 0.0 - 1.0
        dop = dop.to(device, non_blocking=True)
        dop = dop.half() if half else dop.float()  # uint8 to fp16/32
        dop /= 255.0  # 0 - 255 to 0.0 - 1.0

        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        x = [img, aop, dop]

        with torch.no_grad():
            # Run model
            out, train_out = model(x, augment=augment)  # inference and training outputs

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # append json dict
            # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            box = xyxy2xywh(predn[:, :4])  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(pred.tolist(), box.tolist()):
                jdict.append({
                    'image_id': image_id,
                    'category_id': int(p[5]),
                    'bbox': [round(x, 3) for x in b],
                    'score': round(p[4], 5)
                })

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, _ = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%12s' + '%12i' * 2 + '%12.3g' * 3 + '%20.3g'  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Save JSON
    if len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        pred_json = str(save_dir / "predictions.json")  # predictions json
        print('\nSaving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='inference')
    parser.add_argument('--weights', nargs='+', type=str, default='ckpt/PCDNet.pt')
    parser.add_argument('--data', type=str, default='data/rgbpc_coco.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='results', help='path to save prediction results')

    opt = parser.parse_args()

    inference(data=opt.data, weights=opt.weights)
