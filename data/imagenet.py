import os

import torch
from torchvision import datasets, transforms

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

class ImageNet:
    def __init__(self, args):
        super(ImageNet, self).__init__()

        data_root = os.path.join(args.data, "imagenet")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )


        if args.distributed == False:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
            )
        else:
            # For multinode training
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=(self.train_sampler is None), 
                sampler=self.train_sampler,
                **kwargs 
            )

        if args.distributed == False:
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                #shuffle=True,
                **kwargs
            )
        else:
            # For multinode training
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=(self.val_sampler is None), 
                sampler=self.val_sampler,
                **kwargs
            )
