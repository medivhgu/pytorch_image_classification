import torch.utils.data as data

from PIL import Image
import os
import os.path


def make_dataset(dir, list_file):
    images = []
    fin = open(list_file, 'r')
    for readline in fin.readlines():
        path = os.path.join(dir, readline.strip().split(' ')[0])
        item = (path, int(readline.strip().split(' ')[1]))
        images.append(item)
    fin.close()
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    f = open(path, 'rb')
    img = Image.open(f)
    return img.convert('RGB')
    #with open(path, 'rb') as f:
    #    with Image.open(f) as img:
    #        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageLister(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        list_file:
            root/dog/xxx.jpg 0
            root/cat/xxy.jpg 1
            root/dog/12_1.png 0

    Args:
        root (string): Root directory path.
        list_file (string): two column, the first is relative path of image, the second is class_index
        transform (callable, optional): A function/transform that in an PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

    Attributes:
        clsses (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, list_file, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(root, list_file)
        
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index(int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_num_classes(self):
        return max([item[1] for item in self.imgs]) + 1
