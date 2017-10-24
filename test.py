import torchvision_datasets_lister
import sys
import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms

rootdir = sys.argv[1]
list_file = sys.argv[2]

image_lister = torchvision_datasets_lister.ImageLister(rootdir, list_file,
        #transforms.Compose([
        #    transforms.Scale([256, 256]),
        #    transforms.CenterCrop(224),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #])
    )
print(image_lister.get_num_classes())
print(image_lister.__len__())
for item in image_lister.imgs[0:10]:
    print item[0], item[1]
tmp = image_lister.__getitem__(0)
print(tmp)
print(tmp[0].size)


image_folder = datasets.ImageFolder('../CUB_200_2011/images')
print(image_folder.__len__())
for item in image_folder.imgs[0:10]:
    print item[0], item[1]
for item in image_folder.classes[:5]:
    print item, image_folder.class_to_idx[item]
#print image_folder.__getitem__(0)

