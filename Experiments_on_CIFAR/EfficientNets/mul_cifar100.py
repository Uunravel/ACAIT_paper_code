from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data.dataset import ConcatDataset
import cifar


class mul_CIFAR100DataLoader(data.DataLoader):

    def __init__(self, root: str, image_size: int, train: bool, batch_size: int, shuffle: bool, **kwargs):
        #normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))#cifar10
        normalize = transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))#cifar100

        # normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))#cifar100#temp

        if train:
            transform0 = transforms.Compose([
                transforms.RandomCrop(image_size, padding=4),
                # transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            transform1 = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                # transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            transform2 = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4),
                transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0), ratio=( 0.8, 1.2 )),
                # transforms.RandomResizedCrop(image_size,scale=(0.8, 1.2), ratio=( 4./5., 5./4. )),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            transform3 = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                # transforms.RandomResizedCrop(image_size,scale=(0.8, 1.2), ratio=( 4./5., 5./4. )),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        else:
            transform = transforms.Compose([
                # transforms.Resize(int(image_size*1.143)),
                # transforms.Resize(int(256)),
                # transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])
        if train:
            dataset = datasets.CIFAR100(root, train=train, transform=transform0, download=False)
            # dataset = datasets.CIFAR10(root, train=train, transform=transform0, download=False)
            # dataset1 = datasets.CIFAR100(root, train=train, transform=transform1, download=False)
            # dataset2 = datasets.CIFAR100(root, train=train, transform=transform2, download=False)
            # dataset3 = datasets.CIFAR100(root, train=train, transform=transform3, download=False)
            # dataset = ConcatDataset(( dataset1,dataset2, dataset3))
        else:
            dataset = datasets.CIFAR100(root, train=train, transform=transform, download=False)
            #dataset = datasets.CIFAR10(root, train=train, transform=transform, download=False)
        super(mul_CIFAR100DataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
