import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import lmdb
from tqdm import tqdm

# VH, VV
MEAN_S1 = [-19.22, -12.59]
STD_S1 = [5.42, 5.04]


def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class Subset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def random_subset(dataset, frac, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.choice(range(len(dataset)), int(frac * len(dataset)))
    return Subset(dataset, indices)


class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


def make_lmdb(dataset, lmdb_file, num_workers=6):
    loader = InfiniteDataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
    env = lmdb.open(lmdb_file, map_size=1099511627776)

    txn = env.begin(write=True)
    for index, (sample, target) in tqdm(enumerate(loader), total=len(dataset), desc='Creating LMDB'):
        sample = np.array(sample)
        obj = (sample.tobytes(), sample.shape, target.tobytes())
        txn.put(str(index).encode(), pickle.dumps(obj))
        if index % 10000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()

    env.sync()
    env.close()


def make_lmdb_B12_B2(dataset, lmdb_file, num_workers=6):
    loader = InfiniteDataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
    env = lmdb.open(lmdb_file, map_size=1099511627776)

    txn = env.begin(write=True)
    for index, (sample_S2, sample_S1, target) in tqdm(enumerate(loader), total=len(dataset), desc='Creating LMDB'):
        sample_S2 = np.array(sample_S2)
        sample_S1 = np.array(sample_S1)
        obj = (sample_S2.tobytes(), sample_S2.shape, sample_S1.tobytes(), sample_S1.shape, target.tobytes())
        txn.put(str(index).encode(), pickle.dumps(obj))
        if index % 10000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()

    env.sync()
    env.close()


class LMDBDataset_S2(Dataset):

    def __init__(self, lmdb_file, is_slurm_job=False, transform=None):
        self.lmdb_file = lmdb_file
        self.transform = transform
        self.is_slurm_job = is_slurm_job

        if not self.is_slurm_job:
            self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
            with self.env.begin(write=False) as txn:
                self.length = txn.stat()['entries']            
        else:
            # Workaround to have length from the start for ImageNet since we don't have LMDB at initialization time
            self.env = None
            if 'train' in self.lmdb_file:
                self.length = 300000
            elif 'val' in self.lmdb_file:
                self.length = 100000
            else:
                raise NotImplementedError

    def _init_db(self):
        
        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __getitem__(self, index):
        if self.is_slurm_job:
            # Delay loading LMDB data until after initialization
            if self.env is None:
                self._init_db()
        
        with self.env.begin(write=False) as txn:
            data = txn.get(str(index).encode())

        sample_bytes, sample_shape, target_bytes = pickle.loads(data)
        sample = np.fromstring(sample_bytes, dtype=np.uint8).reshape(sample_shape) 
        target = np.fromstring(target_bytes, dtype=np.float32)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return self.length


class LMDBDataset_S1(Dataset):
    def __init__(self, lmdb_file, is_slurm_job=False, transform=None):
        self.lmdb_file = lmdb_file
        self.transform = transform
        self.is_slurm_job = is_slurm_job

        if not self.is_slurm_job:
            self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
            with self.env.begin(write=False) as txn:
                self.length = txn.stat()['entries']            
        else:
            # Workaround to have length from the start for ImageNet since we don't have LMDB at initialization time
            self.env = None
            if 'train' in self.lmdb_file:
                self.length = 300000
            elif 'val' in self.lmdb_file:
                self.length = 100000
            else:
                raise NotImplementedError

    def _init_db(self):
        
        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __getitem__(self, index):
        if self.is_slurm_job:
            # Delay loading LMDB data until after initialization
            if self.env is None:
                self._init_db()
        
        with self.env.begin(write=False) as txn:
            data = txn.get(str(index).encode())

        sample_bytes, sample_shape, target_bytes = pickle.loads(data)
        sample = np.fromstring(sample_bytes, dtype=np.float32).reshape(sample_shape)
        # normalize s1
        self.max_q = np.quantile(sample.reshape(-1,2), 0.99, axis=0) # VH,VV       
        self.min_q = np.quantile(sample.reshape(-1,2), 0.01, axis=0) # VH,VV
        img_bands = []
        for b in range(2):
            img = sample[:,:,b].copy()
            # outlier
            max_q = self.max_q[b]
            min_q = self.min_q[b]            
            img[img>max_q] = max_q
            img[img<min_q] = min_q
            # normalize
            img = normalize(img, MEAN_S1[b], STD_S1[b])       
            img = img.reshape(128, 128, 1)
            img_bands.append(img)
        sample = np.concatenate(img_bands, axis=2)   
        target = np.fromstring(target_bytes, dtype=np.float32)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return self.length


# 加载make_lmdb_B12_B2生成的B12（归一化）+B2（未归一化）的 train_B12_B2.lmdb 文件，并transform预处理后输出，适用Bigearthnet_B12_B2_no_norm生成的数据文件
# 代码出自：https://github.com/zhu-xlab/DINO-MM/tree/main/datasets/BigEarthNet
class LMDBDataset_S1_S2(Dataset):

    def __init__(self, lmdb_file, is_slurm_job=False, transform=None):
        self.lmdb_file = lmdb_file
        self.transform = transform
        self.is_slurm_job = is_slurm_job

        if not self.is_slurm_job:
            self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
            with self.env.begin(write=False) as txn:
                self.length = txn.stat()['entries']            
        else:
            # Workaround to have length from the start for ImageNet since we don't have LMDB at initialization time
            self.env = None
            if 'train' in self.lmdb_file:
                self.length = 300000
            elif 'val' in self.lmdb_file:
                self.length = 100000
            else:
                raise NotImplementedError

    def _init_db(self):
        
        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __getitem__(self, index):
        if self.is_slurm_job:
            # Delay loading LMDB data until after initialization
            if self.env is None:
                self._init_db()
        
        with self.env.begin(write=False) as txn:
            data = txn.get(str(index).encode())

        sample_s2_bytes, sample_s2_shape, sample_s1_bytes, sample_s1_shape, target_bytes = pickle.loads(data)
        sample_s2 = np.fromstring(sample_s2_bytes, dtype=np.uint8).reshape(sample_s2_shape)
        sample_s1 = np.fromstring(sample_s1_bytes, dtype=np.float32).reshape(sample_s1_shape)

        ### normalize s1
        self.max_q = np.quantile(sample_s1.reshape(-1,2), 0.99, axis=0) # VH,VV       
        self.min_q = np.quantile(sample_s1.reshape(-1,2), 0.01, axis=0) # VH,VV
        img_bands = []
        for b in range(2):
            img = sample_s1[:,:,b].copy()
            ## outlier
            max_q = self.max_q[b]
            min_q = self.min_q[b]            
            img[img>max_q] = max_q
            img[img<min_q] = min_q
            ## normalize
            img = normalize(img, MEAN_S1[b], STD_S1[b])       
            img = img.reshape(128, 128, 1)
            img_bands.append(img)
        sample_s1 = np.concatenate(img_bands, axis=2)
        sample = np.concatenate([sample_s2, sample_s1], axis=2)
        target = np.fromstring(target_bytes, dtype=np.float32)        
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return self.length


if __name__ == '__main__':
    import torch
    from cvtorchvision import cvtransforms
    from rs_transforms_float32 import RandomBrightness, RandomContrast, ToGray, GaussianBlur

    seed = 42

    class TwoCropsTransform:
        """Take two random crops of one image"""

        def __init__(self, base_transform1, base_transform2):
            self.base_transform1 = base_transform1
            self.base_transform2 = base_transform2

        def __call__(self, x):
            im1 = self.base_transform1(x)
            im2 = self.base_transform2(x)
            return [im1, im2]

    
    augmentation1 = [
        cvtransforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        cvtransforms.RandomApply([
            RandomBrightness(0.4),
            RandomContrast(0.4)
        ], p=0.8),
        cvtransforms.RandomApply([ToGray(14)], p=0.2),
        cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.ToTensor()
    ]

    augmentation2 = [
        cvtransforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        cvtransforms.RandomApply([
            RandomBrightness(0.4),
            RandomContrast(0.4)
        ], p=0.8),
        cvtransforms.RandomApply([ToGray(14)], p=0.2),
        cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        cvtransforms.RandomHorizontalFlip(),  
        cvtransforms.ToTensor()
    ]   
    train_transforms1 = cvtransforms.Compose(augmentation1)
    train_transforms2 = cvtransforms.Compose(augmentation2)
    
    train_dataset = LMDBDataset_S1_S2(
        lmdb_file='/home/wgd/code/Datasets/BigEarthNet/lmdb_data/val_B12_B2.lmdb',
        transform=TwoCropsTransform(train_transforms1, train_transforms2)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=4)    
    for idx, (img, target) in enumerate(train_loader):
        print(idx)
