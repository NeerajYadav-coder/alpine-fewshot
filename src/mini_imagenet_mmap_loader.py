# mini_imagenet_mmap_loader.py
# Zero-RAM MiniImageNet Loader using memory-mapped disk file.
# Uses np.memmap for 0 MB RAM overhead during training and evaluation.

import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class MiniImageNetMmap:
    """
    Zero-RAM few-shot dataset loader using disk memory mapping (np.memmap).
    Splits 100 total classes into:
      - split='train': first 64 classes (meta-training)
      - split='test':  last 20 classes (evaluation)
    RAM Overhead: ~0 MB (OS handles paging directly from disk).
    """

    def __init__(self, split='train', seed=42):
        assert split in ['train', 'test'], "split must be 'train' or 'test'"
        np.random.seed(seed)

        mmap_dir = 'data/mini_imagenet_mmap'
        mmap_file = os.path.join(mmap_dir, 'images_84x84.npy')
        meta_file = os.path.join(mmap_dir, 'metadata.json')

        if not os.path.exists(mmap_file) or not os.path.exists(meta_file):
            raise FileNotFoundError("MMAP dataset files not found. Run build_mmap_dataset.py first.")

        with open(meta_file, 'r') as f:
            metadata = json.load(f)

        total_imgs = metadata['total_imgs']
        labels = metadata['labels']

        # Memory map the 84x84 uint8 array directly from disk (0 MB RAM allocation)
        self.mmap_data = np.memmap(mmap_file, dtype='uint8', mode='r', shape=(total_imgs, 84, 84, 3))

        # Index indices per class
        class_indices = {}
        for idx, lbl in enumerate(labels):
            if lbl not in class_indices:
                class_indices[lbl] = []
            class_indices[lbl].append(idx)

        sorted_classes = sorted(class_indices.keys())
        
        # Standard MiniImageNet split: 64 train, 20 test
        train_classes = sorted_classes[:64]
        test_classes  = sorted_classes[80:] if len(sorted_classes) >= 100 else sorted_classes[64:84]

        selected_classes = train_classes if split == 'train' else test_classes
        self.class_keys = list(selected_classes)

        self.class_indices = {k: class_indices[k] for k in self.class_keys}
        total_samples = sum(len(v) for v in self.class_indices.values())

        print(f"[MiniImageNetMmap] {split.upper()} split ready: {len(self.class_keys)} classes, "
              f"{total_samples} samples. RAM overhead: 0 MB (memory-mapped).", flush=True)

    def _get_tensor(self, idx):
        # Convert uint8 (84, 84, 3) np.memmap slice to PyTorch normalized tensor
        arr = self.mmap_data[idx]  # HxWx3 uint8
        img = Image.fromarray(arr)
        return TRANSFORM(img)

    def sample_episode(self, n_way=5, k_shot=1, q_query=15, seed=None):
        if seed is not None:
            np.random.seed(seed)

        chosen_classes = np.random.choice(self.class_keys, n_way, replace=False)
        sx, sy, qx, qy = [], [], [], []

        for mapped_label, cls_id in enumerate(chosen_classes):
            indices = self.class_indices[cls_id]
            chosen = np.random.choice(indices, k_shot + q_query, replace=False)

            for idx in chosen[:k_shot]:
                sx.append(self._get_tensor(idx))
                sy.append(mapped_label)

            for idx in chosen[k_shot:]:
                qx.append(self._get_tensor(idx))
                qy.append(mapped_label)

        return (torch.stack(sx), torch.tensor(sy, dtype=torch.long),
                torch.stack(qx), torch.tensor(qy, dtype=torch.long),
                chosen_classes.tolist())
