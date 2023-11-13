# Originally found in https://github.com/lucidrains/DALLE-pytorch
import argparse
import clip
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule


import numpy as np
import torch
import deeplake
from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from deeplake.enterprise.dataloader import indra_available

class ConvertRGBAtoRGB:
    def __call__(self, img):
        if img.mode == 'RGBA':
            # Convert RGBA to RGB
            img = img.convert('RGB')
        return img

def image_transform(img):
    if np.array( img ).shape[0] == 1:
      return torch.tensor([]) # Mark corrupted images, these will be removed in collate_fn

    transform = T.Compose([T.ToPILImage(),
                           ConvertRGBAtoRGB(),
                           T.ToTensor(),
                           T.RandomResizedCrop(224, scale=(
                               0.75, 1.), ratio=(1., 1.)),
                           T.Normalize((0.48145466, 0.4578275, 0.40821073),
                                       (0.26862954, 0.26130258, 0.27577711))
                           ])
    t = transform(img)
    return t


def txt_transform(txt: str, custom_tokenizer=False):
    if custom_tokenizer:
      return txt
    else:
      t = clip.tokenize(txt, truncate=True)[0].numpy()
      return t


def collate_fn(batch, custom_tokenizer=False):

    elem = batch[0]

    if isinstance(elem, IterableOrderedDict):
        key_value_pairs = []
        for key in elem.keys():
          values = []
          for d in batch:
            if d['URL'].numel() != 0: # Remove corrupted images, the empty tensors (look at "image_transform" func)
              values.append(d[key])

          processed_values = collate_fn(values)

          key_value_pairs.append((key, processed_values))

        return IterableOrderedDict(key_value_pairs)

    if custom_tokenizer:
        tokens = custom_tokenizer(
            [row['TEXT'] for row in batch], padding=True, truncation=True, return_tensors="pt")
        batch = [(row[0], token) for row, token in zip(batch, tokens)]

    if isinstance(elem, np.ndarray) and elem.dtype.type is np.str_:
        batch = [it.item() for it in batch]

    return torch.utils.data._utils.collate.default_collate(batch)

class DeepLakeDataModule(LightningDataModule):
    def __init__(self,
                 batch_size: int = 8,
                 num_workers: int = 1,
                 num_threads: int = 1,
                 image_size: int = 224,
                 resize_ratio: int = 0.75,
                 shuffle: bool = False,
                 path: str = None,
                 token: str = None,
                 custom_tokenizer: bool = False
                 ):
        """Create a text image datamodule from directories with congruent text and image names.

        Args:
            batch_size (int): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader. Defaults to 0.
            image_size (int, optional   ): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            path (str, optional): deep lake dataset dataset containing image and caption tensors matched. Defaults to None.
            token (str, optional): deep lake dataset token. Defaults to None.
            custom_tokenizer (bool, optional): Whether or not there is a custom tokenizer. Defaults to False.
        """
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.image_size = image_size
        self.rezie_ratio = resize_ratio
        self.custom_tokenizer = custom_tokenizer
        self.ds = deeplake.load(path, token=token)

        if filter_NSFW:
          print("Filtering NSFW records... (It might take a while)")
          self.ds = self.ds.query("SELECT * WHERE NSFW=='UNLIKELY'")
          print("Filtering Done.")

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int,
                            help='size of the batch', default=64)
        parser.add_argument('--num_workers', type=int, default=2,
                            help='number of workers for the dataloaders')
        parser.add_argument('--image_size', type=int,
                            default=224, help='size of the images')
        parser.add_argument('--path', type=str,
                            default="s3://...", help='size of the images')
        parser.add_argument('--token', type=str, default="",
                            help='token for accessing the dataset (optional)')
        return parser

    def setup(self, stage=None):
        print(self.batch_size)
        print("Dataset retrieved")

    def train_dataloader(self):
        custom_tokenizer = self.custom_tokenizer

        def txt_transform_local(txt): return txt_transform(
            txt, custom_tokenizer=custom_tokenizer)

        def update_collate_fn(batch): return collate_fn(
            batch, custom_tokenizer=custom_tokenizer)

        if indra_available():
            # Fast dataloader implemented in CPP
            self.train_dl = self.ds.dataloader() \
              .transform({'URL': image_transform, 'TEXT': txt_transform_local})\
              .batch(self.batch_size, drop_last=True)\
              .pytorch(num_workers=self.num_workers, collate_fn=update_collate_fn)
        else:
            # For Windows machines where the CPP implementation is not available.
            self.train_dl = self.ds.pytorch(
                num_workers=self.num_workers,
                transform={'URL': image_transform,
                           'TEXT': txt_transform_local},
                batch_size=self.batch_size,
                tensors=['URL', 'TEXT'],
                shuffle=self.shuffle,
                drop_last=True,
                collate_fn=update_collate_fn)

        return self.train_dl
