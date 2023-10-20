import os
import math
import torch

from PIL import Image, ImageDraw
from typing import Optional
from ultralytics import YOLO

from lib.glass_defect_dataset.src.datasets.glass_plate import SinglePlate, GlassPlateTrainYolo, GlassPlate
from src.utils.config_parser import Config
from src.utils.tools import Logger


class YoloTrain:

    P2 = "-p2"
    SAVE_IMG_SIZE = 256

    def __init__(self, config: Config):
        self.config = config

    def execute(self, dataset: Optional[GlassPlate]=None):
        if self.config.train_test.model_test_path is None:
            Logger.instance().debug(f"No pre-trained model found. Must train yolo first")
            self.train_test()
        else:
            Logger.instance().debug(f"Specified pre-trained model at {self.config.train_test.model_test_path}")
            self.test()
    
    def train_test(self):
        """Perform both train and test with ultralytics' yolov8

        Since val parameter is True by default, it will train and test yolov8 on the selected dataset

        SeeAlso:
            https://docs.ultralytics.com/usage/cfg/#train
            https://github.com/ultralytics/ultralytics/issues/981
            https://docs.ultralytics.com/models/yolov8/#supported-modes
        """

        Logger.instance().debug(f"Performing yolo train and test")
        
        yolo_model_name = self.config.model.model_name #"yolov8s"
        if self.P2 in yolo_model_name:
            model = YOLO(f"{yolo_model_name}-p2.yaml").load(f"{yolo_model_name.split(self.P2)[0]}.pt")
        else:
            model = YOLO(f"{yolo_model_name}.pt")
        
        model.train(
            task="detect",
            data=os.path.join(self.config.dataset.dataset_path, "data.yaml"),
            name=os.path.join(os.getcwd(), "output", f"{yolo_model_name}"),
            imgsz=self.config.dataset.image_size,
            epochs=self.config.train_test.epochs,
            batch=self.config.train_test.batch_size,
            degrees=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            mixup=0.0,
            copy_paste=0.0,
            flipud=0.0,
            fliplr=0.0,
            #hsv_h=0.0,
            #hsv_s=0.0,
            #hsv_v=0.0
            translate=0.0,
            mosaic=0.0
        )

    def test(self):
        """Perform test on patches that have already been stored
        
        This function saves the patches in the output folder drawing the detection rectangle (red) together with the
        ground truth (red), if exists.
        """
        
        Logger.instance().debug(f"Testing yolo")

        det_color = (255,0,0)
        gth_color = (0,255,0)

        model = YOLO(self.config.train_test.model_test_path)
        results = model.predict(f"{os.path.join(self.config.dataset.dataset_path, 'test', 'images')}", stream=True)

        tot_defect_gt = 0
        tot_defect_det = 0
        false_positive = 0
        false_negative = 0
        correct_det = 0

        for result in results:
            img_jpg = Image.fromarray(result.orig_img.astype("uint8"), "RGB")
            draw = ImageDraw.Draw(img_jpg)
            curr_img_name = os.path.basename(result.path)
            out_path = f"output/{curr_img_name}"

            boxes_list = torch.round(result.boxes.xyxy).type(torch.int64).tolist()
            boxes_conf = result.boxes.conf.tolist()
            boxes_cls = torch.round(result.boxes.cls).type(torch.int64).tolist()

            # detections
            if len(boxes_list) > 0:
                tot_defect_det += len(boxes_list)
                for box in range(len(boxes_list)):
                    left, top, right, bottom = boxes_list[box]
                    draw.rectangle((left, top, right, bottom), outline=det_color, width=1)
                    draw.text((left, top-20), f"{boxes_conf[box]:.2f}", fill=det_color)
                img_jpg.save(out_path)    

            # ground truth
            gt_path = os.path.join(self.config.dataset.dataset_path, "test", "labels", f"{curr_img_name.replace('.png', '.txt')}")
            defects_gt = GlassPlateTrainYolo.read_annotations_back(gt_path)
            if defects_gt is not None:
                tot_defect_gt += len(defects_gt)
                for defect in defects_gt:
                    l = math.floor((SinglePlate.UPSCALE/SinglePlate.PATCH_SIZE) * defect.min_x)
                    t = math.floor((SinglePlate.UPSCALE/SinglePlate.PATCH_SIZE) * defect.min_y)
                    r = math.ceil((SinglePlate.UPSCALE/SinglePlate.PATCH_SIZE) * defect.max_x)
                    b = math.ceil((SinglePlate.UPSCALE/SinglePlate.PATCH_SIZE) * defect.max_y)
                    draw.rectangle((l, t, r, b), outline=gth_color, width=1)
                img_jpg.save(out_path)

            # count
            true_def = 0 if defects_gt is None else len(defects_gt)
            if len(boxes_list) > true_def:
                false_positive += (len(boxes_list) - true_def)
            elif true_def > len(boxes_list):
                false_negative += (true_def - len(boxes_list))
            else:
                correct_det += true_def

        Logger.instance().debug(
            f"total defects: {tot_defect_gt}, total detections: {tot_defect_det}, false positives: {false_positive}, " +
            f"false negatives: {false_negative} " +
            f"correct detections: {correct_det}"
        )


class YoloInference:

    SAVE_IMG_SIZE = 256

    def __init__(self, config: Config):
        self.config = config
        
    def execute(self, dataset: GlassPlate):
        
        Logger.instance().debug(f"Testing yolo")

        det_color = (255,0,0)
        gth_color = (0,255,0)

        model = YOLO(self.config.train_test.model_test_path)

        tot_defect_gt = 0
        tot_defect_det = 0
        false_positive = 0
        false_negative = 0
        correct_det = 0

        for plate in dataset.filtered_plates:
            plate_id = os.path.basename(plate.ch_1.split("_1.png")[0])
            for batch_idx, (batch_patch, batch_patch_pil) in enumerate(dataset.analyze_plate(plate)):
                patch = batch_patch
                patch_pil = batch_patch_pil
                results = model.predict(patch_pil, stream=True)

                for patch_idx, result in enumerate(results):
                    img_jpg = Image.fromarray(result.orig_img.astype("uint8"), "RGB").resize((self.SAVE_IMG_SIZE, self.SAVE_IMG_SIZE))
                    draw = ImageDraw.Draw(img_jpg)
                    img_path = f"output/plate_{plate_id}_patch_{batch_idx}_b{patch_idx}.jpg"

                    boxes_list = torch.round(result.boxes.xyxy).type(torch.int64).tolist()
                    boxes_conf = result.boxes.conf.tolist()
                    boxes_cls = torch.round(result.boxes.cls).type(torch.int64).tolist()

                    # detections
                    if len(boxes_list) > 0:
                        tot_defect_det += len(boxes_list)
                        for box in range(len(boxes_list)):
                            left, top, right, bottom = boxes_list[box]
                            left = math.floor(left * self.SAVE_IMG_SIZE / SinglePlate.UPSCALE)
                            top = math.floor(top * self.SAVE_IMG_SIZE / SinglePlate.UPSCALE)
                            right = math.ceil(right * self.SAVE_IMG_SIZE / SinglePlate.UPSCALE)
                            bottom = math.ceil(bottom * self.SAVE_IMG_SIZE / SinglePlate.UPSCALE)
                            draw.rectangle((left, top, right, bottom), outline=det_color, width=1)
                            draw.text((left, top-20), f"{boxes_conf[box]:.2f}", fill=det_color)
                        img_jpg.save(img_path)    

                    # ground truth
                    if patch[patch_idx].defects is not None:
                        tot_defect_gt += len(patch[patch_idx].defects)
                        for defect in patch[patch_idx].defects:
                            l = round((self.SAVE_IMG_SIZE/SinglePlate.PATCH_SIZE) * defect.min_x)
                            t = round((self.SAVE_IMG_SIZE/SinglePlate.PATCH_SIZE) * defect.min_y)
                            r = round((self.SAVE_IMG_SIZE/SinglePlate.PATCH_SIZE) * defect.max_x)
                            b = round((self.SAVE_IMG_SIZE/SinglePlate.PATCH_SIZE) * defect.max_y)
                            draw.rectangle((l, t, r, b), outline=gth_color, width=1)
                        img_jpg.save(img_path)

                    # count
                    true_def = 0 if patch[patch_idx].defects is None else len(patch[patch_idx].defects)
                    if len(boxes_list) > true_def:
                        false_positive += (len(boxes_list) - true_def)
                    elif true_def > len(boxes_list):
                        false_negative += (true_def - len(boxes_list))
                    else:
                        correct_det += true_def

        Logger.instance().debug(
            f"total defects: {tot_defect_gt}, total detections: {tot_defect_det}, false positives: {false_positive}, "\
            f"false negatives: {false_negative} "\
            f"correct detections: {correct_det}"
        )
