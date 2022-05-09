import torch
from torchvision import transforms, datasets

class TwoCrargsransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def set_loader(args):
    # construct data loader
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif args.dataset == 'path':
        mean = eval(args.mean)
        std = eval(args.std)
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.data_folder,
                                         transform=TwoCrargsransform(train_transform),
                                         download=True)
        test_dataset = datasets.CIFAR10(root=args.data_folder,
                                        transform=TwoCrargsransform(test_transform),
                                        download=True)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=args.data_folder,
                                          transform=TwoCrargsransform(train_transform),
                                          download=True)
        test_dataset = datasets.CIFAR100(root=args.data_folder,
                                        transform=TwoCrargsransform(transforms.Compose(transforms.ToTensor(), normalize)),
                                        download=True)
    elif args.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=args.data_folder,
                                            transform=TwoCrargsransform(train_transform))
        test_dataset = None
    else:
        raise ValueError(args.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, sampler=None)

    return train_loader, test_loader