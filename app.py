import datetime
from typing import List
from functools import lru_cache
import config

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn
from BboxToolkit import obb2poly, obb2hbb

import torch
import mmcv
from mmcv.parallel import collate, scatter
from mmdet.apis import init_detector
from mmdet.apis.obb.huge_img_inference import LoadPatch, parse_split_cfg, get_windows, merge_patch_results
from mmdet.datasets.pipelines import Compose



def create_image_info(
    image_id,
    file_name,
    image_size,
    date_captured=datetime.datetime.utcnow().isoformat(" "),
    license_id=1,
    coco_url="",
    flickr_url="",
):

    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url,
    }

    return image_info


def create_annotation_info(annotation_id, image_id, category_info, obbox, score):

    hbbox = obb2hbb(obbox).tolist()
    hbbox = [hbbox[0], hbbox[1], hbbox[2] - hbbox[0], hbbox[3] - hbbox[1]]

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": 0,
        "area": hbbox[2] * hbbox[3],
        "bbox": hbbox,
        "segmentation": [obb2poly(obbox).tolist()],
        "score": float(score),
    }

    return annotation_info


def create_category_info(supercategory, id, name):
    category_info = {"supercategory": supercategory, "id": id, "name": name}

    return category_info


meta_info = {
    "year": 2021,
    "version": "1.0",
    "description": "Ship detection",
    "contributor": "",
    "url": "via",
    "date_created": "",
}

@lru_cache()
def get_settings():
    return config.Settings()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ("ship",)


settings = config.Settings()

model_0_5m = init_detector(settings.config_path_0_5m, settings.cpkt_path_0_5m, device=settings.device)
model_3m = init_detector(settings.config_path_3m, settings.cpkt_path_3m, device=settings.device)
split_cfg_3m = parse_split_cfg(settings.split_cfg_3m)
split_cfg_0_5m = parse_split_cfg(settings.split_cfg_0_5m)

def load_image_into_numpy_array(data):
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def detect_huge_image(model, img, split_cfg, merge_cfg):
    """append annotations per class to result dict
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadPatch()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    img = mmcv.imread(img)
    height, width = img.shape[:2]
    sizes, steps = split_cfg
    windows = get_windows(width, height, sizes, steps)
    # detection loop
    results = []
    prog_bar = mmcv.ProgressBar(len(windows))
    for win in windows:
        data = dict(img=img)
        data['patch_win'] = win.tolist()
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [device])[0]
        if device != 'cpu':
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = data['img_metas'][0].data

        # forward the model
        with torch.no_grad():
            results.append(model(return_loss=False, rescale=True, **data))
        prog_bar.update()
    # merge results
    results = merge_patch_results(results, windows, merge_cfg)
    return results


@app.post("/detectship")
async def upload_file(files: List[UploadFile] = File(...), model_type: str = Form(...)):
    res = {"info": meta_info, "images": [], "annotations": [], "categories": []}

    #model: DetectorModel
    if model_type == "0_5m":
        model = model_0_5m
        split_cfg = split_cfg_3m
    elif model_type == "3m":
        model = model_3m
        split_cfg = split_cfg_3m
    else:
        return { "error": "specify model_type: 0_5m or 3m" }
    try:
        nms_cfg = dict(type='BT_nms', iou_thr=0.5)
        for i, name in enumerate(CLASSES):
            res["categories"].append(create_category_info(name, i + 1, name))
        image_id = 1
        annotation_id = 1
        for file in files:
            img = load_image_into_numpy_array(await file.read())
            height, width, _ = img.shape
            # detections = model.inference_single(img, (1024, 1024), (2048, 2048))

            detections = detect_huge_image(model, img, split_cfg, nms_cfg)
            res["images"].append(
                create_image_info(image_id, file.filename, (width, height))
            )
            if len(detections) != 0:
                bboxes = np.vstack(detections)
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(detections)
                ]
                labels = np.concatenate(labels)
                bboxes, scores = bboxes[:, :-1], bboxes[:, -1]
                bboxes = bboxes[scores > settings.score_thr]
                labels = labels[scores > settings.score_thr]
                scores = scores[scores > settings.score_thr]
                bboxes = np.concatenate([bboxes, scores[:, None]], axis=1)
                bboxes = [bboxes[labels == i] for i in range(labels.max()+1)]
                for i, cls_bboxes in enumerate(bboxes):
                    cls_bboxes, cls_scores = cls_bboxes[:, :-1], cls_bboxes[:, -1]
                    for j in range(len(cls_bboxes)):
                        ann = create_annotation_info(
                            annotation_id, image_id, res["categories"][i], cls_bboxes[j], cls_scores[j]
                        )
                        res["annotations"].append(ann)
                        annotation_id += 1

            image_id += 1

    except Exception as e:
        print(e)
    return res


if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping
    uvicorn.run(app, port=8080, host="0.0.0.0")
