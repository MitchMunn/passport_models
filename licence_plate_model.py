import math
import yolov5
import cv2
from yolov5.utils.dataloaders import exif_transpose, letterbox
from yolov5.utils.general import make_divisible, non_max_suppression, scale_boxes
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import requests
from dataset_models import AIPassportTable
import dataset_models
import utils
from datasets import load_dataset
import polars as pl
from pathlib import Path
import os
# Model: https://huggingface.co/keremberke/yolov5n-license-plate


class AukusYoloV5Adaptor(nn.Module):
    def __init__(self, model_name, M, N, iou=0.45, conf=0.25, size=640, agnostic=False, classes=None) -> None:
        """ Adapts the inputs and outputs for a YoloV5 Object detection model

        Args:
            model_name: the model name
            M: The number of bounding boxes per prediction
            N: number of class predictions made per bounding box
            iou: intersection over union
            conf: confidence threshold
            classes: filter by class, i.e. = [0, 15] for cats and dog
            agnostic: NMS class-agnostic
        
        Returns:
            None
        """
        super(AukusYoloV5Adaptor, self).__init__()
        self.model = yolov5.load(model_name)
        self.M = M
        self.N = N
        self.size = size

        self.model.max_det = M
        if N <= 1:
            self.model.multi_label = False
        else:
            self.model.multi_label = True
        self.model.iou = iou
        self.model.conf = conf
        self.model.agnostic = False
        self.model.classes = None
        self.model.eval()

        self.max_det = M
        if N <= 1:
            self.multi_label = False
        else:
            self.multi_label = True
        self.iou = iou
        self.conf = conf
        self.agnostic = agnostic 
        self.classes = classes 


    def forward(self, x):
        """ Forward pass to perform inference and adapt YOLOv5 outputs.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, W, H).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - boxes_tensor (B, M, 4): Bounding box coordinates.
                - scores_tensor (B, M, N): Probability distributions over classes.
                - classes_tensor (B, M): Class indices.
        """
        device = self.device
        x.to(self.device)
        results = self.model(x, size=self.size, augment=False)
        results = non_max_suppression(
            results,
            self.conf,
            self.iou,
            self.classes,
            self.agnostic,
            self.multi_label,
            max_det=self.max_det,
        )
        batch_size = x.shape[0]
        # for i in range(batch_size):
        #     scale_boxes((x.size[2], x.size[2]), results[i][:, :4], (x.size[3], x.size[3]))

        boxes_tensor = torch.zeros(
            (batch_size, self.M, 4), dtype=torch.int32, device=device
        )
        scores_tensor = torch.zeros(
            (batch_size, self.M, self.N), dtype=torch.float32, device=device
        )
        classes_tensor = torch.zeros(
            (batch_size, self.M), dtype=torch.int16, device=device
        )

        for i in range(batch_size):
            prediction = results[i]  # (num_detections, 6) tensor: [x1, y1, x2, y2, conf, class]
            num_detections = prediction.shape[0]

            # Limit the number of detections to M
            if num_detections > self.M:
                prediction = prediction[: self.M]
                num_detections = self.M

            if num_detections > 0:
                boxes = prediction[:, :4]
                boxes_tensor[i, :num_detections, :] = boxes
                scores = prediction[:, 4]
                classes = prediction[:, 5].long()
                if self.N == 1:
                    scores_tensor[i, :num_detections, 0] = scores
                    classes_tensor[i, :num_detections] = classes
                else:
                    # Create one-hot encoding for classes
                    one_hot = torch.zeros(
                        (num_detections, self.N), dtype=torch.float32, device=device
                    )
                    one_hot.scatter_(1, classes.unsqueeze(1), scores.unsqueeze(1))
                    scores_tensor[i, :num_detections, :] = one_hot
                    classes_tensor[i, :num_detections] = classes

        return boxes_tensor, scores_tensor, classes_tensor


class AIPassportLicencePlateDatasetAdaptor:
    def __init__(self, dataset_name="keremberke/license-plate-object-detection"):
        self.dataset_name = dataset_name

    def download_data(self):
        ds = load_dataset(self.dataset_name, "full")
        df = self._convert_dataset_to_polars(ds)
        df.write_parquet(file=self._output_table_path())

    def read_dataset(self):
        file_path = self._output_table_path()
        self.dataset = pl.read_parquet(file_path)
        return self.dataset

    def _convert_dataset_to_polars(self, dataset):
        records = []
        for row in dataset["train"]:
            image_id = row["image_id"]
            image_type = "jpg"
            pil_image = row["image"]
            image_width, image_height = pil_image.size
            file_path = self._create_local_file_path(image_id)
            self._write_local_data(file_path, pil_image)
            image = dataset_models.Image(
                file_name=str(file_path), image_type=image_type
            )

            targets = []
            for i in range(len(row["objects"]["bbox"])):
                obj = row["objects"]
                bbox = obj["bbox"][i]
                x1, x2, y1, y2 = map(int, bbox)

                pixel1 = utils.convert_coords_to_pixel(x1, y1, image_width)
                pixel2 = utils.convert_coords_to_pixel(x2, y2, image_width)

                target = dataset_models.TargetItem(
                    target_id=obj["category"][i],
                    bounding_box_top_left_pixel_number=pixel1,
                    bounding_box_bottom_right_pixel_number=pixel2,
                )
                targets.append(target)

            aipassport_record = AIPassportTable(image=image, target=targets)
            records.append(aipassport_record)

        df = pl.DataFrame(records)
        return df

    
    def _create_local_file_path(self, image_id: str):
        os.makedirs(
            f"data/{self.dataset_name}/resources", exist_ok=True
        )
        file_path = Path(f"data/{self.dataset_name}/resources", str(image_id) + ".jpg")
        return file_path


    def _output_table_path(self):
        file_path = Path(f"data/{self.dataset_name}", "table.parquet")
        return file_path


    def _write_local_data(self, file_path: Path, pil_image: Image):
        pil_image.save(file_path, format="JPEG")


# def load_and_convert_to_PIL_image(row):
#     file_name = Path(row['image'].struct.field('file_name').item())
#     img = Image.open(file_name)
#     img = img.resize((640, 640))
#     return img


# def load_and_convert_to_tensor(row) -> torch.Tensor:
#     file_name = Path(row['image'].struct.field('file_name').item())
#     img = Image.open(file_name).convert('RGB')
#     img = img.resize((640, 640))
#     img = np.array(img).astype(np.uint8)
#     img = torch.from_numpy(img).permute(2, 0, 1)
#     return img


def prepare_batches(df: pl.DataFrame, batch_size: int, n_batches: int = None) -> torch.Tensor:
    if n_batches == None:
        n_batches = math.ceil(df.shape[0]/batch_size)

    batches = []
    for i in range(n_batches):
        if i != math.ceil(n_batches):
            df_slice = df.slice(i*batch_size, (i+1)*batch_size)
        else:
            df_slice = df.slice(i*batch_size, df.shape[0])

        # convert batches to tensors format
        batch = yolov5_preprocess([row['image']['file_name'] for row in df_slice.iter_rows(named=True)])
        batches.append(batch)
    return batches


def yolov5_preprocess(ims: list[Image.Image|str|Path], stride=1, size=(640, 640)):
    n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
    shape0, shape1, files = [], [], []  # image and inference shapes, filenames
    for i, im in enumerate(ims):
        f = f"image{i}"  # filename
        if isinstance(im, (str, Path)):  # filename or uri
            im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im), im
            im = np.asarray(exif_transpose(im))
        elif isinstance(im, Image.Image):  # PIL Image
            im, f = np.asarray(exif_transpose(im)), getattr(im, "filename", f) or f
        files.append(Path(f).with_suffix(".jpg").name)
        if im.shape[0] < 5:  # image in CHW
            im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
        im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
        s = im.shape[:2]  # HWC
        shape0.append(s)  # image shape
        g = max(size) / max(s)  # gain
        shape1.append([int(y * g) for y in s])
        ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
    shape1 = [make_divisible(x, stride) for x in np.array(shape1).max(0)]  # inf shape
    x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
    x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
    x = torch.from_numpy(x) / 255  # uint8 to fp16/32
    return(x)


if __name__ == "__main__":
    adaptor = AukusYoloV5Adaptor(
        model_name="keremberke/yolov5n-license-plate", M=3, N=1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adaptor.device = device
    dataset = AIPassportLicencePlateDatasetAdaptor(dataset_name="keremberke/license-plate-object-detection")
    data = dataset.read_dataset()
    batches = prepare_batches(data, batch_size=4, n_batches=1)
    adaptor.to(device)
    boxes, scores, classes = adaptor(batches[0])

    # Display results
    print("Boxes Tensor:", boxes)
    print("Scores Tensor:", scores)
    print("Classes Tensor:", classes)



