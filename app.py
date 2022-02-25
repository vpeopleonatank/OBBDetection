from typing import List
from functools import lru_cache
import config

from rasterio.io import MemoryFile
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn
from BboxToolkit import obb2poly

import torch
import mmcv
from mmcv.parallel import collate, scatter
from mmdet.apis import init_detector
from mmdet.apis.obb.huge_img_inference import (
    LoadPatch,
    parse_split_cfg,
    get_windows,
    merge_patch_results,
)
from mmdet.datasets.pipelines import Compose


def convert_pixel_to_geo(bbox, tf_matrix):
    """Convert pixel coordinates to geography coordinates

    Args:
        bbox (): list of points. Ex: [300, 400, 500, 600]
        tf_matrix (): transform matrix to convert pixel coords to geo coords

    Returns:
        geography coordinates
    """
    geo_pixels = []
    for i in range(0, len(bbox), 2):
        geo_pixels.append([bbox[i], bbox[i + 1]] * tf_matrix)
    geo_pixels.append(geo_pixels[0])

    return geo_pixels


def create_poly_obj_geojson(obbox, score, tf_matrix):

    poly_obj = {
        "type": "Feature",
        "properties": {
            "score": str(float(score)),
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [convert_pixel_to_geo(obb2poly(obbox).tolist(), tf_matrix)],
        },
    }

    return poly_obj


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


# =========================================== Model ====================================================
model_05m_bien = init_detector(
    settings.config_05m_bien, settings.cpkt_05m_bien, device=settings.device
)
model_05m_cang = init_detector(
    settings.config_05m_cang, settings.cpkt_05m_cang, device=settings.device
)
model_05m_dao = init_detector(
    settings.config_05m_dao, settings.cpkt_05m_dao, device=settings.device
)

model_3m_bien = init_detector(
    settings.config_3m_bien, settings.cpkt_3m_bien, device=settings.device
)
model_3m_cang = init_detector(
    settings.config_3m_cang, settings.cpkt_3m_cang, device=settings.device
)
model_3m_dao = init_detector(
    settings.config_3m_dao, settings.cpkt_3m_dao, device=settings.device
)

# =========================================== Split Config ====================================================
split_config_05m_bien = parse_split_cfg(settings.split_config_05m_bien)
split_config_05m_dao = parse_split_cfg(settings.split_config_05m_dao)
split_config_05m_cang = parse_split_cfg(settings.split_config_05m_cang)

split_config_3m_bien = parse_split_cfg(settings.split_config_3m_bien)
split_config_3m_dao = parse_split_cfg(settings.split_config_3m_dao)
split_config_3m_cang = parse_split_cfg(settings.split_config_3m_cang)


def load_content_from_bytes(data):
    with MemoryFile(data) as memfile:
        with memfile.open() as dataset:
            data_array = dataset.read([3, 2, 1])
            img = np.moveaxis(data_array, 0, -1)
            tf_matrix = dataset.transform

            return img, tf_matrix


def detect_huge_image(model, img, split_cfg, merge_cfg):
    """append annotations per class to result dict"""
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build themodel_0_5m data pipeline
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
        data["patch_win"] = win.tolist()
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [device])[0]
        if device != "cpu":
            data = scatter(data, [device])[0]
        else:
            data["img_metas"] = data["img_metas"][0].data

        # forward the model
        with torch.no_grad():
            results.append(model(return_loss=False, rescale=True, **data))
        prog_bar.update()
    # merge results
    results = merge_patch_results(results, windows, merge_cfg)
    return results


@app.post("/detectship")
async def upload_file(
    files: List[UploadFile] = File(...),
    model_type: str = Form(...),
    area_type: str = Form(...),
):
    res = {"type": "FeatureCollection", "features": []}

    # model: DetectorModel
    if model_type == "0.5m" and area_type == "bien":
        model = model_05m_bien
        split_cfg = split_config_05m_bien
    elif model_type == "0.5m" and area_type == "cang":
        model = model_05m_cang
        split_cfg = split_config_05m_cang
    elif model_type == "0.5m" and area_type == "dao":
        model = model_05m_dao
        split_cfg = split_config_05m_dao
    elif model_type == "3m" and area_type == "bien":
        model = model_3m_bien
        split_cfg = split_config_3m_bien
    elif model_type == "3m" and area_type == "cang":
        model = model_3m_cang
        split_cfg = split_config_3m_cang
    elif model_type == "3m" and area_type == "dao":
        model = model_3m_dao
        split_cfg = split_config_3m_dao
    else:
        return {
            "error": "specify model_type: 0.5m or 3m; and area_type: bien or cang or dao"
        }
    try:
        nms_cfg = dict(type="BT_nms", iou_thr=0.5)
        annotation_id = 1
        for file in files:
            img, tf_matrix = load_content_from_bytes(await file.read())

            detections = detect_huge_image(model, img, split_cfg, nms_cfg)
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
                bboxes = [bboxes[labels == i] for i in range(labels.max() + 1)]
                for _, cls_bboxes in enumerate(bboxes):
                    cls_bboxes, cls_scores = cls_bboxes[:, :-1], cls_bboxes[:, -1]
                    for j in range(len(cls_bboxes)):
                        ann = create_poly_obj_geojson(
                            cls_bboxes[j],
                            cls_scores[j],
                            tf_matrix
                        )
                        res["features"].append(ann)
                        annotation_id += 1

    except Exception as e:
        print(e)
    return res


if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping
    uvicorn.run(app, port=8081, host="0.0.0.0")
