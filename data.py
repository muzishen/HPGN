from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler, RandomIdentitySampler
from opt import opt
import os
import re
import torchvision
#torchvision.set_image_backend('accimage')

class Data():
    def __init__(self):
        transform_list = [
            # transforms.Resize((opt.h, opt.w), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(opt.pad, padding_mode='edge'),
            transforms.RandomCrop((opt.h, opt.w)),
        ]
        if opt.cj > 0:
            transform_list.append(
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)]))

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=opt.mean, std=opt.std))
        if opt.erasing_p > 0:
            transform_list.append(RandomErasing(probability=opt.erasing_p, mean=opt.mean))
        print(transform_list)
        train_transform = transforms.Compose(transform_list)

        test_transform = transforms.Compose([
            # transforms.Resize((opt.h, opt.w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean, std=opt.std)
        ])

        self.trainset = Vehicle_Reid(train_transform, 'train', opt.data_path,)
        self.testset = Vehicle_Reid(test_transform, 'test', opt.data_path, )
        self.queryset = Vehicle_Reid(test_transform, 'query', opt.data_path, )

        if opt.sampler == 0:
            self.train_loader = dataloader.DataLoader(self.trainset,batch_size=opt.batchid * opt.batchimage, num_workers=8, shuffle=True, pin_memory=True,drop_last=True)
        elif opt.sampler == 0.5:
            self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                                                        batch_image=opt.batchimage),
                                                  batch_size=opt.batchid * opt.batchimage, num_workers=8,
                                                  pin_memory=True)
        elif opt.sampler == 1:
            self.train_loader = dataloader.DataLoader(self.trainset,
                                                      sampler=RandomIdentitySampler(self.trainset,batch_size=opt.batchimage*opt.batchid,
                                                                                    num_instances=opt.batchimage),
                                                      batch_size=opt.batchid * opt.batchimage, num_workers=8,
                                                      pin_memory=True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=8, pin_memory=True)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=8, pin_memory=True)

        if opt.mode == 'vis':
            self.query_image = test_transform(default_loader(opt.query_image))


class Vehicle_Reid(dataset.Dataset):
    def __init__(self, transform, mode, data_path):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if mode == 'train':
            self.data_path += '/train_all'
        elif mode == 'test':
            self.data_path += '/gallery800/gallery_0'
        else:
            self.data_path += '/probe800/probe_0'

        self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}   # 文件夹标签转成从0开始的标签

    def __getitem__(self, index):
        path = self.imgs[index]
        Vehicle_label = self.id(path)
        target = self._id2label[Vehicle_label]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: single image path    0000006_1.jpg
        :return: vehicle id      88
        """
        #print(file_path.split('/')[-1])
        #return int(file_path.split('/')[-1].split('_')[1].split('.')[0])
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: single image path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1:4])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique vehicle ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])


if __name__ == '__main__':

    data = Data()