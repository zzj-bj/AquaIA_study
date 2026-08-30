import os
from pathlib import Path
from typing import List
from detection.utils.import_utils import (
    fn,
    pipeline_def,
    types,
    ndd,
    DALIRaggedIterator,
    LastBatchPolicy,
)
from PIL import Image

import numpy as np
import torch
import random
from detection.utils.config_utils import load_class_names


# TODO : AutoAugment: Learning Augmentation Strategies from Data
# Z: AutoAugment automatically searches for the best augmentation policies
# Z: transform a regular Python function into a definition function for DALI data processing pipeline
@pipeline_def
def create_detection_pipeline(dataset_src, stats, img_size=640, device="gpu"):
    """Z: DALI data preprocessing pipeline, used to read, decode, resize, and normalize JPG images
    then output them for training. One image treatement pipeline.
    DALI decode, resize, mean/std normalize, PyTorch CHW tensor (inputs).
    Returns inputs, labels, boxes on GPU, idx on CPU.
    """
    # Z: external_source() gets data from outside of DALI pipeline
    # Z: here dataset_src = self.dataset.__call__ = JpgDALIDataset.__call__
    # Z: and JpgDALIDataset.__call__() returns a tuple of (encoded_img, labels, boxes, idx)
    # Z: encoded = encoded image, labels = class labels, boxes = bbox coords, idx = index of image in dataset
    encoded, labels, boxes, idx = fn.external_source(
        source=dataset_src,
        # Z: external source returns 4 outputs
        num_outputs=4,
        # Z: external source returns one sample at a time not a batch
        batch=False,
        # Z: external source can be called in parallel by multiple threads
        parallel=True,
        dtype=[types.UINT8, types.INT64, types.FLOAT, types.INT64],
        # Z: nb dim of each output
        ndim=[1, 1, 2, 1],
    )
    # Z: mixed = read/prepare on CPU, decode on GPU
    decoding_device = "mixed" if device == "gpu" else device
    # TODO : add cache/padding to the decoding part to avoid memory re-allocation
    # Z: decode images to RGB, images on GPU after decoding
    images = fn.decoders.image(encoded, device=decoding_device, output_type=types.RGB)
    images = fn.resize(
        images,
        resize_x=img_size,
        resize_y=img_size,
        device=device,
    )
    # Z: transform to float, mean/std normalize, from HWC to CHW, on GPU
    inputs = fn.crop_mirror_normalize(
        images,
        device=device,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=stats["mean"],
        std=stats["std"],
    )
    # Z: move labels and boxes to GPU
    if device == "gpu":
        labels = labels.gpu()
        boxes = boxes.gpu()
    # Z: inputs is DALI tensor
    return inputs, labels, boxes, idx


class BaseDetectionDataset:
    """
    Base class shared by non-DALI and DALI dataset implementations.
    This class handles dataset metadata and target loading:
    - loads normalization statistics from stats.npy and scales them from [0, 1] to [0, 255] pixel units
    - loads class names and number of classes
    - builds a sorted list of label files
    - parses YOLO-format label files into class labels and bounding boxes
    - stores targets as torch tensors rather than DALI tensors
    - returns cloned targets to avoid modifying the cached source targets
    - no CPU/CUDA/GPU transfer in this base class

    For the non-DALI path, it also provides helpers to:
    - convert numpy images to torch tensors
    - normalize image tensors using dataset statistics
    """

    def __init__(
        self,
        dataset_root: str,
        data_split: str = "train",
        stats_file: str = "stats.npy",
        device: str = "cpu",
    ):
        self.dataset_root = Path(dataset_root)
        self.stats_file = stats_file
        self.data_split = data_split
        self.img_dir = self.dataset_root / "images" / self.data_split
        self.load_stats()
        self.class_names, self.num_classes = load_class_names(dataset_root)
        self.device = device
        self.load_targets()

        # if not (self.dataset_root / self.data_split).exists():
        #     raise FileNotFoundError(f"Data split directory not found: {self.dataset_root / self.data_split}")

    def __len__(self):
        # Z: self.target_files apprears after load_targets() is called
        return len(self.target_files)

    def load_stats(self) -> None:
        """Z: load dataset mean/std statistics and scale them from [0, 1] to [0, 255]
        for normalization of decoded JPG pixels."""
        # /!\ Expect stats to be computed in normalized pixels in [0, 1] range
        stats_path = self.dataset_root / self.stats_file

        if stats_path.exists():
            # Z: allow_pickle=True allows loading Python objects like dict, .npy file may be a dict
            # Z: .item() transforms to a dict
            stats = np.load(stats_path, allow_pickle=True).item()
            self.stats = {
                "mean": stats["mean"] * np.float32(255.0),
                "std": np.clip(
                    stats["std"],
                    min=1e-6,
                )
                * np.float32(255.0),
            }
        else:
            raise FileNotFoundError(f"Stats file not found: {stats_path}")

    def to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Z: np [H,W,C] -> pytorch [C,H,W]. Only for non-DALI situations."""
        # Z: not called in actual class
        return torch.from_numpy(img).float().permute(2, 0, 1)

    @staticmethod
    # Z: static method don't need class param
    def _numeric_sort_key(path: Path):
        """Z: generate a sort key for file paths,
        purely numeric filenames are sorted by their numerical value,
        non-numeric filenames follow numeric ones and are sorted alphabetically."""
        # Z: get file name without extension
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    def get_sorted_target_files(self) -> List[str]:
        target_dir = self.dataset_root / "labels" / self.data_split
        target_files = [path for path in target_dir.glob("*.txt")]
        return sorted(target_files, key=self._numeric_sort_key)

    def parse_target_line(self, line: str):
        class_id, x_center, y_center, width, height = line.split()
        return int(class_id), [float(x_center), float(y_center), float(width), float(height)]

    def read_target(self, label_path: str):
        """Z: read one label file and parse targets (class labels and bbox coords)."""
        labels = []
        boxes = []
        if label_path is not None and os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    class_id, bbox = self.parse_target_line(line)
                    labels.append(class_id)
                    boxes.append(bbox)
        # TODO : clean up, dict struct is not longer necessary
        return {
            "labels": torch.tensor(labels, dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        }

    def load_targets(self) -> None:
        """Z: load label files, parse targets, store as torch tensors.
        A dict per sample with keys "labels" and "boxes". A list of dicts for all samples."""
        self.target_files = self.get_sorted_target_files()
        if not self.target_files:
            raise FileNotFoundError(f"No label files found under {self.dataset_root / 'labels' / self.data_split}")
        self.targets = [self.read_target(path) for path in self.target_files]

    def copy_target(self, idx: int) -> dict:
        """Z: clone and return target (label+bbox) at an index to avoid modifying source target.
        A dict per sample with keys "labels" and "boxes"."""
        # Z: not called in actual class
        return {key: value.clone() for key, value in self.targets[idx].items()}

    def normalize_img(self, img: torch.Tensor) -> torch.Tensor:
        """Z: mean/std normalize image tensor. Only for non-DALI situations."""
        # Z: not called in actual class
        # Z: transform mean and std to torch tensors and reshape to [C, 1, 1] for broadcasting
        mean = torch.from_numpy(self.stats["mean"]).to(dtype=img.dtype).view(-1, 1, 1)
        std = torch.from_numpy(self.stats["std"]).to(dtype=img.dtype).view(-1, 1, 1)
        return (img - mean) / std


class JpgDALIDataset(BaseDetectionDataset):
    """Z: Mainly used for DALI's external_source, which means the following __call__
    will be repeatedly called by DALI to read image bytes. One image.
    Train (__call__): JPG bytes, numpy uint8 buffer (encoded_img). Decode resize normalize will be done in DALI pipeline.
    Non-train (__getitem__): JPG bytes, DALI decode, resize, PyTorch CHW tensor (sample["image"]).
    Non-train (__getitem__): JPG bytes, DALI decode, resize, mean/std normalize, PyTorch CHW tensor (sample["input"])."""
    # TODO : only JPEG, need to think about TIFF handling

    def __init__(
        self,
        dataset_root: str,
        data_split: str = "train",
        batch_size: int = 16,
        img_size: int = 640,
        img_format: str = "jpg",
        stats_file: str = "stats.npy",
        device: str = "cpu",
    ):
        super().__init__(
            dataset_root=dataset_root,
            data_split=data_split,
            stats_file=stats_file,
            device=device,
        )
        # Z: after called super().__init__(), we have self.img_dir, self.stats, self.class_names,
        # self.num_classes, self.target_files, self.targets
        self.img_size = img_size
        self.batch_size = batch_size
        self.img_format = img_format
        if img_format not in ["jpg", "jpeg"]:
            raise NotImplementedError(f"Unsupported image format: {img_format}. Only jpg is currently supported.")

        self.n = len(self.target_files)
        # Z: create indices for all samples
        self.indices = list(range(self.n))
        # Z: compute the number of full batches
        self.full_iterations = self.n // batch_size
        # Shuffling related stuff
        self.perm = self.indices  # permutation of indices
        # Z: last_seen_epoch is used to track the epoch index for shuffling
        # Z: all samples in the same epoch have same self.perm
        self.last_seen_epoch = (
            # so that we don't have to recompute the `self.perm` for every sample
            None
        )

    @staticmethod
    def _dali_tensor_to_torch(tensor):
        """Z: convert DALI tensor to PyTorch tensor."""
        return torch.from_dlpack(tensor.evaluate().data)

    def __call__(self, sample_info):
        """Z: called by DALI's external_source to get a sample (encoded image bytes, labels, boxes, index)."""
        # Z: get sample's position in actual epoch from sample_info given by DALI
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration
        if self.data_split == "train":
            # Shuffling at the start of each epoch
            if self.last_seen_epoch != sample_info.epoch_idx:
                self.last_seen_epoch = sample_info.epoch_idx
                # Z: create a random number generator
                self.perm = np.random.default_rng(seed=42 + sample_info.epoch_idx)
                # Z: shuffle the indices for this epoch
                self.perm = self.perm.permutation(self.indices)
        # Z: find the true dataset index based on the current sample's position in the epoch
        idx = self.perm[sample_idx]
        # Z: get image file name without extension
        img_id = self.target_files[idx].stem
        img_path = self.img_dir / f"{img_id}.{self.img_format}"
        # Encoded image bytes. DALI will decode this on the GPU.
        # Z: read original binary image bytes from disk and convert to numpy array of uint8
        encoded_img = np.frombuffer(img_path.read_bytes(), dtype=np.uint8)
        target = self.targets[idx]
        return (
            encoded_img,
            target["labels"].numpy(),
            target["boxes"].numpy(),
            np.array([idx], dtype=np.int64),
        )

    def __getitem__(self, key):
        # Slow but useful for sampling a few images for visualization / testing
        # Mirrors the DALI pipeline path using DALI dynamic operators.
        # Z: find the true dataset index
        idx = self.indices[key]
        # Z: get image file name without extention
        img_id = self.target_files[idx].stem
        img_path = self.img_dir / f"{img_id}.{self.img_format}"

        # Z: read original binary image bytes from disk and convert to numpy array of uint8
        # Z: .copy() is used to ensure that the numpy array has its own memory
        encoded_img = np.frombuffer(img_path.read_bytes(), dtype=np.uint8).copy()
        device = "gpu" if self.device == "cuda" else "cpu"
        decoding_device = "mixed" if device == "gpu" else device
        mean = self.stats["mean"].astype(np.float32).tolist()
        std = self.stats["std"].astype(np.float32).tolist()

        # Z: decode images to RGB
        img = ndd.decoders.image(encoded_img, device=decoding_device, output_type=types.RGB)
        img = ndd.resize(
            img,
            resize_x=float(self.img_size),
            resize_y=float(self.img_size),
            device=device,
        )
        # Z: transform to float, mean/std normalize, from HWC to CHW
        norm_img = ndd.crop_mirror_normalize(
            img,
            device=device,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=mean,
            std=std,
        )
        # Z: convert DALI tensor to CHW PyTorch tensor, on CPU
        img = self._dali_tensor_to_torch(img).cpu().permute(2, 0, 1)
        # Z: norm_img is already in CHW format, on GPU
        norm_img = self._dali_tensor_to_torch(norm_img)
        sample = {
            "image": img,
            "input": norm_img,
            "target": self.copy_target(idx),
            "target_idx": idx,
            "img_path": str(img_path),
        }
        return sample


class JpgDetectionDataset(BaseDetectionDataset):
    """Z: This class is used for non-DALI situations, where images are loaded and processed using PIL and NumPy.
    One image. JPG bytes, PIL decode, resize, CHW pytorch tensor (sample["image"]),
    mean/std normalize (sample["input"]). No CPU/CUDA/GPU transfer in this class."""
    def __init__(
        self,
        dataset_root: str,
        img_size: int = 640,
        stats_file: str = "stats.npy",
        device: str = "cpu",
        data_split: str = "train",
    ):
        super().__init__(dataset_root=dataset_root, stats_file=stats_file, data_split=data_split, device=device)
        self.img_size = (img_size, img_size)

    def __len__(self) -> int:
        return len(self.target_files)

    def __getitem__(self, idx: int):
        img_id = self.target_files[idx].stem
        img_path = self.img_dir / f"{img_id}.jpg"
        img = Image.open(img_path).convert("RGB").resize(self.img_size)
        img = np.array(img, dtype=np.float32)
        img = self.to_tensor(img)
        norm_img = self.normalize_img(img)
        sample = {
            "image": img,
            "input": norm_img,
            "target": self.copy_target(idx),
            "target_idx": idx,
            "img_path": str(img_path),
        }
        return sample


class DALIDetectionDataLoader:
    """Z: wrap JpgDALIDataset into a DALI dataloader for training loop as `for batch in loader`"""
    def __init__(
        self,
        dataset,
        device="gpu",  # can be dropped and inferred from dataset, but keeping it explicit for now
        # Z: nb threads used for DALI pipeline execution
        num_threads=3,
        # Z: nb Python workers used by DALI when calling a Python external_source
        py_num_workers=3,
        py_start_method="spawn",
    ):
        self.dataset = dataset
        self.device = device
        self.pipeline = create_detection_pipeline(
            dataset_src=self.dataset.__call__,
            stats=self.dataset.stats,
            device=self.device,
            img_size=self.dataset.img_size,
            batch_size=self.dataset.batch_size,
            num_threads=num_threads,
            py_num_workers=py_num_workers,
            py_start_method=py_start_method,
        )
        self.pipeline.build()
        self.loader = DALIRaggedIterator(
            pipelines=[self.pipeline],
            # Z: define outputs, batch["inputs"] etc
            output_map=["inputs", "labels", "boxes", "targets_idx"],
            # Z: declare output types for each output, DALI tensor or list of DALI tensors
            output_types=[
                DALIRaggedIterator.DENSE_TAG,
                DALIRaggedIterator.SPARSE_LIST_TAG,
                DALIRaggedIterator.SPARSE_LIST_TAG,
                DALIRaggedIterator.DENSE_TAG,
            ],
            # Z: nb samples per epoch
            size=self.dataset.full_iterations * self.dataset.batch_size,
            # Z: drop last batch if smaller than batch_size
            last_batch_policy=LastBatchPolicy.DROP,
            # Z: reset iterator
            auto_reset=True,
        )

    def __len__(self):
        # Z: return full batch nb
        return self.dataset.full_iterations

    def __iter__(self):
        # Z: define iteration behavior
        for batch in self.loader:
            if isinstance(batch, list):
                batch = batch[0]
            # Z: .pop() removes the key from the dict and returns its value
            labels_batch = batch.pop("labels")
            boxes_batch = batch.pop("boxes")
            # Z: zip() pairs each labels and boxes from the batch together
            batch["targets"] = [{"labels": labels, "boxes": boxes} for labels, boxes in zip(labels_batch, boxes_batch)]
            # Z: yield means that this function is a generator, it will return a batch and pause until the next call to __next__()
            # Z: { "inputs": ..., "targets_idx": ...,
            # Z: "targets": [ {"labels": ..., "boxes": ...}, {"labels": ..., "boxes": ...}, ... ], }
            yield batch


def parse_batch(batch, device=None):
    """Z: extract model inputs and targets from a dataloader batch
    and convert targets to a per-image list of dictionaries on the specified device.
    If non DALI, moves labels and boxes to GPU."""
    if isinstance(batch, list):
        batch = batch[0]
    inputs = batch["inputs"]
    targets = batch["targets"]

    # Z: non DALI, detection_collate_fn() returns targets as a dict with keys "labels", "boxes", "counts"
    if isinstance(targets, dict):
        labels = targets["labels"]
        boxes = targets["boxes"]
        if device is not None:
            # Z: move labels (all in one tensor) and boxes (all in one tensor) to device
            labels = labels.to(device, non_blocking=True)
            boxes = boxes.to(device, non_blocking=True)
        # Z: split labels and boxes into per-image lists based on counts
        labels_per_image = labels.split(targets["counts"])
        boxes_per_image = boxes.split(targets["counts"])
        # Z: reconstruct targets as a list of dicts, one per image, with keys "labels" and "boxes"
        targets = [{"labels": image_labels, "boxes": image_boxes} for image_labels, image_boxes in zip(labels_per_image, boxes_per_image)]
    # Z: DALI
    elif device is not None:
        # Z: move each value in target dict to device if it's a tensor, otherwise keep it as is
        targets = [{key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in target.items()} for target in targets]
    # Z: inputs = torch.Tensor( shape=[B, 3, H, W], dtype=torch.float32, )
    # Z: targets = [ { "labels": torch.Tensor( shape=[N_i], dtype=torch.int64, ),
    # Z:                "boxes": torch.Tensor( shape=[N_i, 4], dtype=torch.float32, ), }, ... ]
    return inputs, targets


def sample_indices(dataset_size, num_samples, seed):
    """Z: randomly sample some sample indices from the dataset and return them sorted."""
    rng = random.Random(seed)
    sample_size = min(num_samples, dataset_size)
    return sorted(rng.sample(range(dataset_size), sample_size))


def detection_collate_fn(batch):
    """Z: stack single sample into batch. Only for non-DALI situations.
    No CPU/CUDA/GPU transfer."""
    # Z: nb targets per image may vary, later use target_counts to re-split concatenated labels and boxes.
    target_counts = [len(item["target"]["labels"]) for item in batch]
    collated_batch = {
        # Z: [B, 3, H, W]
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "inputs": torch.stack([item["input"] for item in batch], dim=0),
        "targets": {
            # Z: concatenate the class labels of all images in the batch into a 1D tensor, [Nb targets of the batch]
            "labels": torch.cat([item["target"]["labels"] for item in batch], dim=0),
            # Z: concatenate the bbox of all images in the batch into a 2D tensor, [Nb targets of the batch, 4]
            "boxes": torch.cat([item["target"]["boxes"] for item in batch], dim=0),
            # Z: list of nb targets per image, [B]
            "counts": target_counts,
        },
        # Z: [indices]
        "targets_idx": [item["target_idx"] for item in batch],
        # Z: [paths]
        "img_paths": [item["img_path"] for item in batch],
    }
    return collated_batch


def sample_dataset(dataset, num_samples, seed, device):
    """Z: randomly sample from dataset, return model input batch, visualization batch, image paths.
    samples = {"inputs": inputs, "images": imgs, "img_paths": img_paths}."""
    sampled_indices = sample_indices(len(dataset), num_samples, seed)
    # Z: get samples
    samples = [dataset[index] for index in sampled_indices]
    inputs = torch.stack([sample["input"] for sample in samples], dim=0).to(device)
    imgs = torch.stack([sample["image"] for sample in samples], dim=0)
    img_paths = [sample["img_path"] for sample in samples]
    samples = {"inputs": inputs, "images": imgs, "img_paths": img_paths}
    return samples

"""
==================
Output information
==================

DALI training path:
JpgDALIDataset.__call__ -> encoded_img, labels, boxes, idx
create_detection_pipeline -> inputs, labels, boxes, idx
DALIDetectionDataLoader -> inputs, targets (labels, boxes), targets_idx in batch

DALI __getitem__ path:
JpgDALIDataset.__getitem__ -> image, input, target (labels, boxes), target_idx, img_path

non-DALI path:
JpgDetectionDataset.__getitem__ -> image, input, target (labels, boxes), target_idx, img_path
detection_collate_fn -> images, inputs, targets (labels, boxes, counts), targets_idx, img_paths in batch
"""