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
    DALI decode, resize, mean/std normalize, PyTorch CHW tensor (inputs)
    """
    # Z: external_source() gets data from outside of DALI pipeline
    # Z: here dataset_src = self.dataset.__call__ = JpgDALIDataset.__call__
    # Z: and JpgDALIDataset.__call__() returns a tuple of (encoded_img, idx)
    # Z: encoded = encoded image, idx = index of the image in the dataset
    encoded, idx = fn.external_source(
        source=dataset_src,
        # Z: num_outputs=2 means the external source returns two outputs, encoded image and index
        num_outputs=2,
        # Z: external source returns one sample at a time not a batch
        batch=False,
        # Z: external source can be called in parallel by multiple threads
        parallel=True,
        dtype=[types.UINT8, types.INT64],
    )
    # Z: mixed = read/prepare on CPU, decode on GPU
    decoding_device = "mixed" if device == "gpu" else device
    # TODO : add cache/padding to the decoding part to avoid memory re-allocation
    # Z: decode images to RGB
    images = fn.decoders.image(encoded, device=decoding_device, output_type=types.RGB)
    images = fn.resize(
        images,
        resize_x=img_size,
        resize_y=img_size,
        device=device,
    )
    # Z: transform to float, mean/std normalize, from HWC to CHW
    inputs = fn.crop_mirror_normalize(
        images,
        device=device,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=stats["mean"],
        std=stats["std"],
    )
    # Z: inputs is DALI tensor
    return inputs, idx


class BaseDetectionDataset:
    """
    Base class shared by NPY / PIL / RAM datasets.

    Handles:
    - label loading
    - statistics (mean/std)
    - normalization

    Z: This class handles dataset metadata and target loading:
    - loads normalization statistics from stats.npy converted to [0 - 255]
    - loads class names and number of classes
    - builds a sorted list of label files
    - parses YOLO-format label files into class labels and bounding boxes
    - stores targets as torch tensors rather than DALI tensors

    For the non-DALI path, it also provides helpers to:
    - convert numpy images to torch tensors
    - normalize image tensors using dataset statistics

    Subclasses are responsible for image loading and path-specific preprocessing.
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
        # Load targets to device directly to avoid repeated memcpy
        # We do not need to think about targets device at all after this11
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
        """Z: read label files and parse targets (class labels and bbox coords)."""
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
            "labels": torch.tensor(labels, dtype=torch.int64).to(self.device),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4).to(self.device),
        }

    def load_targets(self) -> None:
        """Z: load label files, parse targets, store as tensors on the specified device."""
        self.target_files = self.get_sorted_target_files()
        if not self.target_files:
            raise FileNotFoundError(f"No label files found under {self.dataset_root / 'labels' / self.data_split}")
        self.targets = [self.read_target(path) for path in self.target_files]

    def get_targets(self, batch) -> List[dict]:
        """Z: get targets (label bbox) for a batch of samples accroding to image indices.
        Used by DALI and non-DALI situations because only "targets_idx" is returned by batches."""
        # Z: not called in actual class
        # Z: DALIRaggedIterator sometimes returns a list, where the first element is the actual batch dict
        if isinstance(batch, list):
            batch = batch[0]
        # Z: DALI pipeline only returns DALI tensors and indices
        # Z: return labels and bounding boxes for each image
        return [self.targets[idx] for idx in batch["targets_idx"]]

    def normalize_img(self, img: torch.Tensor) -> torch.Tensor:
        """Z: normalize image tensor to zero mean and unit variance using dataset statistics.
        Only for non-DALI situations."""
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
        """Z: called by DALI's external_source to get a sample (encoded image bytes and index)."""
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
        # Z: get image file name without extention
        img_id = self.target_files[idx].stem
        img_path = self.img_dir / f"{img_id}.{self.img_format}"
        # Encoded image bytes. DALI will decode this on the GPU.
        # Z: read original binary image bytes from disk and convert to numpy array of uint8
        encoded_img = np.frombuffer(img_path.read_bytes(), dtype=np.uint8)
        return encoded_img, np.array([idx])

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

        img = ndd.decoders.image(encoded_img, device=decoding_device, output_type=types.RGB)
        img = ndd.resize(
            img,
            resize_x=float(self.img_size),
            resize_y=float(self.img_size),
            device=device,
        )
        # Z: transform to float, normalize to zero mean and unit variance, and change layout from HWC to CHW
        norm_img = ndd.crop_mirror_normalize(
            img,
            device=device,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=mean,
            std=std,
        )
        # Z: convert DALI tensor to CHW PyTorch tensor
        img = self._dali_tensor_to_torch(img).cpu().permute(2, 0, 1)
        # Z: norm_img is already in CHW format
        norm_img = self._dali_tensor_to_torch(norm_img)
        sample = {
            "image": img,
            "input": norm_img,
            "target_idx": idx,
            "img_path": str(img_path),
        }
        return sample


class JpgDetectionDataset(BaseDetectionDataset):
    """Z: This class is used for non-DALI situations, where images are loaded and processed using PIL and NumPy.
    One image.
    JPG bytes, PIL decode, resize, CHW pytorch tensor (sample["image"]), mean/std normalize (sample["input"])."""
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
        # tgt = self.targets[idx]
        sample = {
            "image": img,
            "input": norm_img,
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
            # Z: define outputs, batch["inputs"] et batch["targets_idx"]
            output_map=["inputs", "targets_idx"],
            # Z: declare two outputs are dense, each sample in a batch has same shape can be stacked into a tensor
            output_types=[
                DALIRaggedIterator.DENSE_TAG,
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
        return iter(self.loader)


def parse_batch(batch):
    """Z: Extract the model input "inputs" from "batch" outputted by dataloader
    and also extract the img_paths if any."""
    if isinstance(batch, list):
        batch = batch[0]
    inputs = batch["inputs"]
    # TODO : ugly but currently required. Need to modify downstream code to avoid this conversion
    # targets = [{"labels": labels, "boxes": boxes} for labels, boxes in zip(batch["labels"], batch["bboxes"])]
    return inputs, batch.get("img_paths", None)


def sample_indices(dataset_size, num_samples, seed):
    """Z: randomly sample some sample indices from the dataset and return them sorted"""
    rng = random.Random(seed)
    sample_size = min(num_samples, dataset_size)
    return sorted(rng.sample(range(dataset_size), sample_size))


def detection_collate_fn(batch):
    """Z: stack single sample into batch. Only for non-DALI situations."""
    collated_batch = {
        # Z: [B, 3, H, W]
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "inputs": torch.stack([item["input"] for item in batch], dim=0),
        # Z: [indices]
        "targets_idx": [item["target_idx"] for item in batch],
        # Z: [paths]
        "img_paths": [item["img_path"] for item in batch],
    }
    return collated_batch


def sample_dataset(dataset, num_samples, seed, device):
    """Z: randomly sample from dataset, return model input batch, visualization batch, image paths."""
    sampled_indices = sample_indices(len(dataset), num_samples, seed)
    # Z: get samples
    samples = [dataset[index] for index in sampled_indices]
    inputs = torch.stack([sample["input"] for sample in samples], dim=0).to(device)
    imgs = torch.stack([sample["image"] for sample in samples], dim=0)
    img_paths = [sample["img_path"] for sample in samples]
    samples = {"inputs": inputs, "images": imgs, "img_paths": img_paths}
    return samples
