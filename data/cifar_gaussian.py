import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args import args


# Class for adding Gaussian noise during transform in CIFAR10_gaussian
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1, p_noise=0.5):
        self.std = std         # Standard deviation
        self.mean = mean       # Mean
        self.p_noise = p_noise # Probability that noise is added to image (carried out by transforms.RandomApply)
        print('   with ' + self.__repr__())

    # Add noise to the data and clamp so that it is in (0,1) range
    def __call__(self, tensor):
        # Generate noise
        noise = torch.randn(tensor.size()) * self.std + self.mean
        # Add noise to tensor and clamp in range [0,1]
        tensor = torch.clamp(tensor + noise, 0., 1.)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, p_noise={2})'.format(self.mean, self.std, self.p_noise)


class GaussianNoiseDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform Gaussian noise augmentation."""

  def __init__(self, dataset, clean_transform, gaussian_transform, no_jsd=True):
    self.dataset = dataset
    self.clean_transform = clean_transform
    self.gaussian_transform = gaussian_transform
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return self.gaussian_transform(x), y
    else:
      im_tuple = (self.clean_transform(x), self.gaussian_transform(x),
                  self.gaussian_transform(x))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)



# CIFAR10 data loader that adds gaussian noise to a fraction of the data (given by 1-p_clean)
class CIFAR10_gaussian:
    def __init__(self, args):
        super(CIFAR10_gaussian, self).__init__()

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
           transforms.RandomCrop(32, padding=4),
           transforms.RandomCrop(32, padding=4),
           transforms.ToTensor()])
        clean_preprocess = transform=transforms.Compose([normalize])

        gaussian_preprocess = transform=transforms.Compose(
                [transforms.RandomApply([AddGaussianNoise(0., args.std_gauss, 1-args.p_clean)],p=1-args.p_clean),
                 normalize])

        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform,
        )

        train_dataset = GaussianNoiseDataset(train_dataset, clean_preprocess, gaussian_preprocess, no_jsd = (not args.jsd))


        # NEW: For multinode training
        if args.distributed == False:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
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

        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
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
