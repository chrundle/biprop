import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args import args
import data.augmentations as augmentations
import numpy as np

def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.
  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if args.all_augmix_augmentations:
    aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(args.mixture_width):
    image_aug = image.copy()
    depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, args.aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=True):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return aug(x, self.preprocess), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)


class CIFAR10_augmix:
    def __init__(self, args):
        super(CIFAR10_augmix, self).__init__()

        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        
        # Load datasets
        train_transform = transforms.Compose(
          [transforms.RandomHorizontalFlip(),
           transforms.RandomCrop(32, padding=4)])
        preprocess = transforms.Compose(
          [transforms.ToTensor(),
           normalize])
        test_transform = preprocess

        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, transform=test_transform, download=True)
        #base_c_path = './data/cifar/CIFAR-10-C/'
        num_classes = 10

        train_dataset = AugMixDataset(train_dataset, preprocess, no_jsd = (not args.jsd))

        # NEW: For multinode training
        if args.distributed == False:
            self.train_loader = torch.utils.data.DataLoader(
              train_dataset,
              batch_size=args.batch_size,
              shuffle=True,
              **kwargs
            )
        else:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=(self.train_sampler is None), 
                sampler=self.train_sampler,
                **kwargs
            )
        # NEW: For multinode training
        if args.distributed == False:
            self.val_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
            )
        else:
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            self.val_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=(self.val_sampler is None), 
                sampler=self.val_sampler,
                **kwargs
            )

        # Added for testing on CIFAR-10-C dataset
        self.validation_dataset = test_dataset
