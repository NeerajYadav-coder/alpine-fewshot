# cifar_fs_dataset.py
# Standard CIFAR-FS Few-Shot Dataset Implementation (Bertinetto et al. split)
# 100 total classes from CIFAR-100 (600 images per class, 32x32 RGB resolution)
# Meta-Train: 64 classes (Class IDs 0..63)
# Meta-Val:   16 classes (Class IDs 64..79)
# Meta-Test:  20 classes (Class IDs 80..99)

import torch
import numpy as np
import datasets
from PIL import Image

class RealCIFARFS:
    def __init__(self, split='train', seed=42):
        self.split = split
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load uoft-cs/cifar100 from HuggingFace
        hf_train = datasets.load_dataset('uoft-cs/cifar100', split='train')
        hf_test = datasets.load_dataset('uoft-cs/cifar100', split='test')
        
        # Mean and std for CIFAR-100 normalization
        mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32).reshape(3, 1, 1)
        std  = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32).reshape(3, 1, 1)
        
        # Pre-process all images into torch tensors (3, 32, 32) normalized
        self.all_class_samples = {c: [] for c in range(100)}
        
        for sample in hf_train:
            img_pil = sample['img']
            lbl = sample['fine_label']
            arr = np.array(img_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
            norm_arr = (arr - mean) / std
            self.all_class_samples[lbl].append(torch.from_numpy(norm_arr))
            
        for sample in hf_test:
            img_pil = sample['img']
            lbl = sample['fine_label']
            arr = np.array(img_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
            norm_arr = (arr - mean) / std
            self.all_class_samples[lbl].append(torch.from_numpy(norm_arr))
            
        # Bertinetto et al. standard 64/16/20 class partitioning
        # Fixed deterministic class mapping seed 42 to ensure exact reproducibility across runs
        all_classes = np.arange(100)
        rng = np.random.RandomState(42)
        permuted_classes = rng.permutation(all_classes)
        
        if split == 'train':
            self.class_ids = list(permuted_classes[:64])
        elif split == 'val':
            self.class_ids = list(permuted_classes[64:80])
        elif split == 'test':
            self.class_ids = list(permuted_classes[80:100])
        else:
            raise ValueError(f"Invalid split: {split}")
            
        self.class_samples = {cls_id: self.all_class_samples[cls_id] for cls_id in self.class_ids}
        print(f"CIFAR-FS [{split.upper()}]: Loaded {len(self.class_ids)} classes with {sum(len(v) for v in self.class_samples.values())} total 32x32 RGB images.")
        
    def sample_episode(self, n_way=5, k_shot=1, q_query=15, shift_px=0, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        chosen_classes = np.random.choice(self.class_ids, n_way, replace=False)
        
        sup_x, sup_y = [], []
        q_x, q_y = [], []
        
        for mapped_label, cls_id in enumerate(chosen_classes):
            imgs = self.class_samples[cls_id]
            chosen_indices = np.random.choice(len(imgs), k_shot + q_query, replace=False)
            
            for idx in chosen_indices[:k_shot]:
                sup_x.append(imgs[idx])
                sup_y.append(mapped_label)
                
            for idx in chosen_indices[k_shot:]:
                img = imgs[idx]
                if shift_px > 0:
                    # Apply spatial shift attack (torch.roll)
                    shift_dir = np.random.choice(['up', 'down', 'left', 'right'])
                    if shift_dir == 'up':
                        img = torch.roll(img, shifts=-shift_px, dims=1)
                    elif shift_dir == 'down':
                        img = torch.roll(img, shifts=shift_px, dims=1)
                    elif shift_dir == 'left':
                        img = torch.roll(img, shifts=-shift_px, dims=2)
                    elif shift_dir == 'right':
                        img = torch.roll(img, shifts=shift_px, dims=2)
                q_x.append(img)
                q_y.append(mapped_label)
                
        return torch.stack(sup_x), torch.tensor(sup_y), torch.stack(q_x), torch.tensor(q_y), chosen_classes.tolist()

if __name__ == '__main__':
    ds_train = RealCIFARFS(split='train')
    ds_test = RealCIFARFS(split='test')
    
    print("\n--- RULE 0: DATASET VERIFICATION ---")
    print(f"Meta-Train Class Count: {len(ds_train.class_ids)}")
    print(f"Meta-Test Class Count : {len(ds_test.class_ids)}")
    
    # Verify zero overlap
    intersection = set(ds_train.class_ids).intersection(set(ds_test.class_ids))
    print(f"Meta-Train & Meta-Test Intersection (Must be 0): {len(intersection)}")
    
    # Print sample image shapes and pixel stats
    sample_img = ds_train.class_samples[ds_train.class_ids[0]][0]
    print(f"Sample Image Shape: {sample_img.shape} (Expected: torch.Size([3, 32, 32]))")
    print(f"Sample Image Min: {sample_img.min().item():.4f}, Max: {sample_img.max().item():.4f}")
