"""SynthEVox3D dataset loader for E2V 3D reconstruction.

Dataset: SynthEVox3D (Chen et al. ICVR 2023)
  - 13 ShapeNet categories, 39,739 total / 1,040 Tiny
  - Events: .npz arr_0 (N, 4) float64 [timestamp, x, y, polarity]
  - Voxel GT: .npz arr_0 (32, 32, 32) bool
  - Resolution: ~512x512 DVS sensor

Data flow:
  events (N,4) → rescale → event volume (1, 100, 256, 256)
  → 3D ResNet → 3D U-Net → (1, 32, 32, 32)
"""
import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from .event_representation import build_event_volume


SYNSET_TO_CLASS = {
    '02691156': 'Airplane', '02828884': 'Bench', '02933112': 'Cabinet',
    '02958343': 'Car', '03001627': 'Chair', '03211117': 'Displayer',
    '03636649': 'Lamp', '03691459': 'Speaker', '04090263': 'Rifle',
    '04256520': 'Sofa', '04379243': 'Table', '04401088': 'Telephone',
    '04530566': 'Watercraft',
}
CLASS_TO_IDX = {name: i for i, name in enumerate(SYNSET_TO_CLASS.values())}


class SynthEVox3D(Dataset):
    """SynthEVox3D dataset with 3D volume output.

    Args:
        root_dir: path to event_3d_scan_tiny/ directory
        split_csv: path to CSV split file (synset_id, model_id, train/val/test)
        split: "train" | "val" | "test"
        representation: event representation name
        image_size: output frame spatial size
        num_frames: number of time bins
        augmentation: enable data augmentation
    """
    def __init__(self, root_dir, split_csv, split="train",
                 representation="event_frame_pos", image_size=256,
                 num_frames=100, augmentation=False):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.num_frames = num_frames
        self.representation = representation
        self.augmentation = augmentation

        self.samples = self._load_split(split_csv)

    def _load_split(self, split_csv):
        samples = []
        with open(split_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3 or row[2].strip() != self.split:
                    continue
                synset_id, model_id, _ = row
                class_name = SYNSET_TO_CLASS.get(synset_id, synset_id)
                samples.append({
                    'path': os.path.join(self.root_dir, synset_id, model_id),
                    'class_id': CLASS_TO_IDX.get(class_name, -1),
                    'class_name': class_name,
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_events(self, sample_path):
        event_path = os.path.join(sample_path, 'events.npz')
        events = np.load(event_path)['arr_0'].astype(np.float32)
        return events  # (N, 4): [timestamp, x, y, polarity]

    def _load_voxel(self, sample_path):
        voxel_path = os.path.join(sample_path, 'model_normalized.npz')
        return np.load(voxel_path)['arr_0'].astype(np.float32)  # (32, 32, 32)

    def _rescale_events(self, events):
        """Rescale from 512x512 sensor to target image_size."""
        orig_size = 512.0
        scale = self.image_size / orig_size
        events = events.copy()
        events[:, 1] *= scale  # x
        events[:, 2] *= scale  # y
        return events

    def _augment_events(self, events):
        """Random augmentations: flip x/y, reverse polarity."""
        if not self.augmentation:
            return events
        events = events.copy()
        if random.random() < 0.5:
            events[:, 1] = self.image_size - 1 - events[:, 1]
        if random.random() < 0.5:
            events[:, 2] = self.image_size - 1 - events[:, 2]
        if random.random() < 0.5:
            events[:, 3] *= -1
        return events

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load raw data
        events = self._load_events(sample['path'])
        voxel = self._load_voxel(sample['path'])

        # Rescale coordinates (512 → 256)
        events = self._rescale_events(events)

        # Data augmentation (train only)
        if self.split == "train":
            events = self._augment_events(events)

        # Reorder columns: [timestamp, x, y, p] → [x, y, t, p]
        events_t = torch.from_numpy(events)
        events_xy = events_t[:, [1, 2, 0, 3]]  # [x, y, t, p]

        # Build 3D event volume: (1, num_frames, H, W)
        event_volume = build_event_volume(
            events_xy, self.num_frames, self.image_size, self.representation)

        return {
            'event_volume': event_volume,                 # (1, 100, 256, 256)
            'voxel': torch.from_numpy(voxel).unsqueeze(0),  # (1, 32, 32, 32)
            'class_id': sample['class_id'],
            'class_name': sample['class_name'],
        }


def get_dataloader(root_dir, split_csv, split="train", batch_size=5,
                    representation="event_frame_pos", image_size=256,
                    num_frames=100, augmentation=False, num_workers=4,
                    shuffle=True):
    """Create DataLoader for SynthEVox3D."""
    dataset = SynthEVox3D(
        root_dir=root_dir, split_csv=split_csv, split=split,
        representation=representation, image_size=image_size,
        num_frames=num_frames, augmentation=augmentation,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
        drop_last=(split == "train"),
    )
