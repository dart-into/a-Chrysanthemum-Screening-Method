from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np
import cv2
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from PIL import Image
import time
import math
from torch.nn.parallel import DataParallel

import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

torch.backends.cudnn.benchmark = True
from einops import rearrange
import timm
import one_load
import vit

import re
import torch.nn.functional as F
#from utils import load_state_dict_from_url
from collections import OrderedDict



class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
            img_name = str(os.path.join(self.root_dir,str(self.images_frame.iloc[idx, 0])))
            im = Image.open(img_name).convert('RGB')
            if im.mode == 'P':
                im = im.convert('RGB')
            image = np.asarray(im)
            #image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
            rating = self.images_frame.iloc[idx, 1]
            sample = {'image': image, 'rating': rating}

            if self.transform:
                sample = self.transform(sample)
            return sample
        # except Exception as e:
        #     pass


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return {'image': image, 'rating': rating}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        if h == new_h and w == new_w:
            return {'image': image, 'rating': rating}

        elif h == new_h and w != new_w:
            left = np.random.randint(0, w - new_w)
            image = image[0: new_h,
                left: left + new_w]
            return {'image': image, 'rating': rating}

        elif h != new_h and w == new_w:
            top = np.random.randint(0, h - new_h)
            image = image[top: top + new_h,
                0:  new_w]
            return {'image': image, 'rating': rating}

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'rating': rating}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image, 'rating': rating}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        im = image /1.0#/ 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'rating': rating}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}



class BaselineModel1(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)              #add norm
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        #print('L9 size',out.size())
        out = self.bn3(out)
        #out = self.sig(out)
        #print('L10',out)
        #print('L10 size',out.size())
        # out_a = torch.cat((out_a, out_p), 1)

        # out_a = self.sig(out)
        return out

class Net(nn.Module):
    def __init__(self, net1, vnet, linear):
        super(Net, self).__init__()
        self.Net1 = net1
        self.Vnet = vnet
        self.Linear = linear
        self.trans = nn.Conv2d(512, 768, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
       
   
        x1 = self.Net1(x)  #sa
        #print(x1.shape)
        x1 = self.trans(x1)
        x1 = self.Vnet(x1)
        out = self.Linear(x1)
        
        return out


def computeSpearman(dataloader_valid1, model):
    ratings = []
    predictions = []
    with torch.no_grad():
        cum_loss = 0
        total_iterations = len(dataloader_valid1)
        for data1 in dataloader_valid1:
            inputs1 = data1['image']
            batch_size1 = inputs1.size()[0]
            labels1 = data1['rating'].view(batch_size1, -1)

            #for batch_idx, data in enumerate(dataloader_valid):
            #inputs = data['image']
            #batch_size = inputs.size()[0]
            #labels = data['rating'].view(batch_size, -1)
            # labels = labels / 100.0

            if use_gpu:
                try:
                    inputs1, labels1 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda())
                except:
                    print(inputs1, labels1)
            else:
                inputs1, labels1 = Variable(inputs1), Variable(labels1)

            outputs_a = model(inputs1)
            ratings.append(labels1.float())
            predictions.append(outputs_a.float())

    ratings_i = np.vstack([r.detach().cpu().numpy() for r in ratings])
    predictions_i = np.vstack([p.detach().cpu().numpy() for p in predictions])
    a = ratings_i[:,0]
    b = predictions_i[:,0]
    sp = spearmanr(a, b)[0]
    pl = pearsonr(a,b)[0]
        #print(total)
    return sp, pl

def train_model():
    epochs = 20
    task_num = 5
    noise_num1 = 24
    noise_num2 = 25


    net_1 = one_load.densenetnew(pretrained=False)

    densenet_model = models.densenet121(pretrained = True)
    state_dict = densenet_model.features.state_dict()

    for name in list(state_dict.keys()):
        if name.startswith('denseblock4.'):
            del state_dict[name]
        if name.startswith('norm5.'):
            del state_dict[name]
        #print(list(state_dict.keys()))
    net_1.features.load_state_dict(state_dict)
        
    pretrained_cfg_overlay = {'file': r"/home/user/use_trans/pytorch_model.bin"}
    vit_model = timm.create_model('vit_base_patch16_224', pretrained_cfg_overlay = pretrained_cfg_overlay ,pretrained=True)
    VIT = vit.VisionTransformer()  
    state_dict = vit_model.state_dict()
    VIT.load_state_dict(state_dict)                
    l_net = BaselineModel1(1, 0.5, 1000)
                         
    #net_1 = models.densenet121(pretrained = True)
    model = Net(net1 = net_1, vnet = VIT, linear = l_net)
    '''
    param_names = [name for name, _ in model.named_parameters()]
    with open('param_names_F.txt', 'w') as f:
        for name in param_names:
            f.write("%s\n" % name)
    print("Parameter names saved to param_names.txt")
    '''
    
    #model.load_state_dict(torch.load('model_IQA/TID2013_KADID10K_4_1.pt'))


    criterion = nn.MSELoss()
    ignored_params = list(map(id, model.Linear.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())
    optimizer = optim.Adam([
        {'params': base_params},
        {'params': model.Linear.parameters(), 'lr': 1e-2}
    ], lr=1e-4)
    model.cuda()
    model = DataParallel(model)
    
    meta_model = copy.deepcopy(model)
    temp_model = copy.deepcopy(model)

    spearman = 0

    for epoch in range(epochs):
        running_loss = 0.0
        optimizer = exp_lr_scheduler(optimizer, epoch)

        list_noise = list(range(noise_num1))
        np.random.shuffle(list_noise)
        
        
        print('############# TID 2013 train phase epoch %2d ###############' % epoch)
        count = 0
        for index in list_noise:

            if count % task_num == 0:
                name_to_param = dict(temp_model.named_parameters())
                for name, param in meta_model.named_parameters():
                    diff = param.data - name_to_param[name].data
                    name_to_param[name].data.add_(diff)

            name_to_param = dict(model.named_parameters())
            for name, param in temp_model.named_parameters():
                diff = param.data - name_to_param[name].data
                name_to_param[name].data.add_(diff)

            dataloader_train1, dataloader_valid1 = load_data('train1', 'tid2013', index)         

            if dataloader_train1 == 0:
                continue

            dataiter1 = iter(enumerate(dataloader_valid1))

            model.train()  # Set model to training mode
            # Iterate over data.

            total_iterations = len(dataloader_train1)
            for data1 in dataloader_train1:
                inputs1 = data1['image']
                batch_size1 = inputs1.size()[0]
                labels1 = data1['rating'].view(batch_size1, -1)

                if use_gpu:
                    try:
                        inputs1, labels1 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda())
                    except:
                        print(inputs1, labels1)
                else:
                    inputs1, labels1 = Variable(inputs1), Variable(labels1)

                optimizer.zero_grad()
                outputs = model(inputs1)
                #print('outputs', outputs)
                loss = criterion(outputs, labels1)
                #print('labels1', labels1)
                #print('loss', loss)
                loss.backward()
                optimizer.step()

                idx1, data_val1 = next(dataiter1)
                
                if idx1 >= len(dataloader_valid1)-1:
                    dataiter1 = iter(enumerate(dataloader_valid1))
                    
                inputs_val1 = data_val1['image']
                batch_size1 = inputs_val1.size()[0]
                labels_val1 = data_val1['rating'].view(batch_size1, -1)
                # labels_val = labels_val / 10.0
                
                if use_gpu:
                    try:
                        inputs_val1, labels_val1 = Variable(inputs_val1.float().cuda()), Variable(labels_val1.float().cuda())
                        
                    except:
                        print(inputs_val1, labels_val1)
                else:
                    inputs_val1, labels_val1 = Variable(inputs_val1), Variable(labels_val1)

                optimizer.zero_grad()
                outputs_val = model(inputs_val1)
                loss_val = criterion(outputs_val, labels_val1)
                loss_val.backward()
                optimizer.step()

                try:
                    running_loss += loss_val.item()
                except:
                    print('unexpected error, could not calculate loss or do a sum.')

                name_to_param1 = dict(meta_model.named_parameters())
                name_to_param2 = dict(temp_model.named_parameters())
                for name, param in model.named_parameters():
                    diff = param.data - name_to_param2[name].data
                    name_to_param1[name].data.add_(diff / task_num)

                count += 1
        # print('trying epoch loss')
        epoch_loss = running_loss / count
        print('current loss = ',epoch_loss)


        running_loss = 0.0
        list_noise = list(range(noise_num2))
        np.random.shuffle(list_noise)
        # list_noise.remove(ii)
        print('############# Kadid train phase epoch %2d ###############' % epoch)
        count = 0
        for index in list_noise:
            if count % task_num == 0:
                name_to_param = dict(temp_model.named_parameters())
                for name, param in meta_model.named_parameters():
                    diff = param.data - name_to_param[name].data
                    name_to_param[name].data.add_(diff)

            name_to_param = dict(model.named_parameters())
            for name, param in temp_model.named_parameters():
                diff = param.data - name_to_param[name].data
                name_to_param[name].data.add_(diff)

            #dataloader_train, dataloader_valid = load_data('train', 'kadid10k', index)
            dataloader_train1, dataloader_valid1 = load_data('train1', 'kadid10k', index)


            if dataloader_train1 == 0:
                continue

            dataiter1 = iter(enumerate(dataloader_valid1))

            model.train()  # Set model to training mode
            # Iterate over data.

            total_iterations = len(dataloader_train1)
            for data1 in dataloader_train1:
                inputs1 = data1['image']
                batch_size1 = inputs1.size()[0]
                labels1 = data1['rating'].view(batch_size1, -1)
                # labels = labels / 10.0
                labels1 = (labels1 - 0.5) / 5.0

                if use_gpu:
                    try:
                        inputs1, labels1 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda())
                        
                    except:
                        print(inputs1, labels1)
                else:
                    inputs1, labels1 = Variable(inputs1), Variable(labels1)

                optimizer.zero_grad()
                outputs = model(inputs1)
                loss = criterion(outputs, labels1)
                loss.backward()
                optimizer.step()

                idx1, data_val1 = next(dataiter1)

                if idx1 >= len(dataloader_valid1)-1:
                    dataiter1 = iter(enumerate(dataloader_valid1))

                inputs_val1 = data_val1['image']
                batch_size1 = inputs_val1.size()[0]
                labels_val1 = data_val1['rating'].view(batch_size1, -1)
                labels_val1 = (labels_val1 - 0.5) / 5.0

                if use_gpu:
                    try:
                        inputs_val1, labels_val1 = Variable(inputs_val1.float().cuda()), Variable(labels_val1.float().cuda())
                    except:  
                        print(inputs_val1, labels_val1)
                else:
                    inputs_val1, labels_val1 = Variable(inputs_val1), Variable(labels_val1)


                optimizer.zero_grad()
                outputs_val = model(inputs_val1)
                loss_val = criterion(outputs_val, labels_val1)
                loss_val.backward()
                optimizer.step()

                try:
                    running_loss += loss_val.item()
                except:
                    print('unexpected error, could not calculate loss or do a sum.')

                name_to_param = dict(meta_model.named_parameters())
                for name, param in model.named_parameters():
                    diff = param.data - name_to_param[name].data
                    name_to_param[name].data.add_(diff / task_num)

                count += 1
        # print('trying epoch loss')
        epoch_loss = running_loss / count
        print('current loss = ',epoch_loss)

        

        print('############# test phase epoch %2d ###############' % epoch)
        dataloader_train1, dataloader_valid1 = load_data('test1', 0)

        model.eval()
        model.cuda()
        sp = computeSpearman(dataloader_valid1, model)[0]
        if sp > spearman:
            spearman = sp
            best_model = copy.deepcopy(model)
            #best_model = copy.deepcopy(meta_model)
            # torch.save(best_model.cuda(),
            #        'model_IQA/TID2013_KADID10K_IQA_Meta_resnet18.pt')
        
        '''
        for k,v in model.state_dict().items():
            with open('record.txt', 'a')as file:
                file.write(f"the {epoch} record:{k}{v}\n")
                #file.save()
                file.close()
        '''
        #for param_tensor in model.state_dict():
        #    print(param_tensor, model.state_dict()[param_tensor])
        print('new srocc {:4f}, best srocc {:4f}'.format(sp, spearman))
    torch.save(model.cuda().state_dict(), 
               'model_IQA/TID2013_Kadid_Meta.pt')
    #torch.save(model.cuda(),
    #       'model_IQA/TID2013_KADID10K_IQA_Meta_resnet18.pt')
    

def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=2):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.9**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_data(mod = 'train', dataset = 'tid2013', worker_idx = 0):

    if dataset == 'tid2013':
        data_dir = os.path.join('/home/user/use_trans/tid2013')
        worker_orignal = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_per_noise.csv'), sep=',')
        image_path1 = '/home/user/data/tid2013/distorted_images/'
        image_path2 = '/home/user/data/tid2013/salient_images/'
        image_path3 = '/home/user/data/tid2013/non_salient_images/'
    else:
        data_dir = os.path.join('/home/user/use_trans/kadid10k')
        worker_orignal = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_per_noise.csv'), sep=',')
        image_path1 = '/home/user/data/Kadid/kadid10k/images/'
        image_path2 = '/home/user/data/Kadid/kadid10k/salient_images/'
        image_path3 = '/home/user/data/Kadid/kadid10k/non_salient_images/'
    workers_fold = "noise/"
    if not os.path.exists(workers_fold):
        os.makedirs(workers_fold)

    #worker = worker_orignal['noise'].unique()[worker_idx]
    #print("----worker number: %2d---- %s" %(worker_idx, worker))
    worker = worker_orignal['noise'].unique()[worker_idx]
    
    #print("----worker number: %2d---- %s" %(worker_idx, worker))

    percent = 0.8
    images = worker_orignal[worker_orignal['noise'].isin([worker])][['image', 'dmos']]

    train_dataframe, valid_dataframe = train_test_split(images, train_size=percent)
    train_path = workers_fold + "train_scores_" + str(worker) + ".csv"
    test_path = workers_fold + "test_scores_" + str(worker) + ".csv"
    train_dataframe.to_csv(train_path, sep=',', index=False)
    valid_dataframe.to_csv(test_path, sep=',', index=False)
    
    b_s = 24
    b_s_v = 20
    
    if mod == 'train1':

        print("----worker number: %2d---- %s" %(worker_idx, worker))
        output_size = (224, 224)
        transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                        root_dir=image_path1,
                                                        transform=transforms.Compose([Rescale(output_size=(300, 300)),
                                                                                      RandomHorizontalFlip(0.5),
                                                                                      RandomCrop(output_size=output_size),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                        root_dir=image_path1,
                                                        transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        dataloader_train = DataLoader(transformed_dataset_train, batch_size=b_s, drop_last=True,
                                  shuffle=False, num_workers=0, collate_fn=my_collate)
        dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=b_s_v, drop_last=True,
                                      shuffle=False, num_workers=0, collate_fn=my_collate)

    elif mod == 'train2':
        output_size = (256, 256)
        transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                        root_dir=image_path2,
                                                        transform=transforms.Compose([Rescale(output_size=(300, 300)),
                                                                                      RandomHorizontalFlip(0.5),
                                                                                      RandomCrop(output_size=output_size),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                        root_dir=image_path2,
                                                        transform=transforms.Compose([#RandomCrop(output_size=output_size),
                                                                                      Rescale(output_size=(224, 224)),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        dataloader_train = DataLoader(transformed_dataset_train, batch_size=b_s, drop_last=True,
                                  shuffle=False, num_workers=0, collate_fn=my_collate)
        dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=b_s_v, drop_last=True,
                                      shuffle=False, num_workers=0, collate_fn=my_collate)

    elif mod == 'train3':

        output_size = (256, 256)
        transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                        root_dir=image_path3,
                                                        transform=transforms.Compose([Rescale(output_size=(300, 300)),
                                                                                      RandomHorizontalFlip(0.5),
                                                                                      RandomCrop(output_size=output_size),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                        root_dir=image_path3,
                                                        transform=transforms.Compose([#RandomCrop(output_size=output_size),
                                                                                      Rescale(output_size=(224, 224)),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        dataloader_train = DataLoader(transformed_dataset_train, batch_size=b_s, drop_last=True,
                                  shuffle=False, num_workers=0, collate_fn=my_collate)
        dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=b_s_v, drop_last=True,
                                      shuffle=False, num_workers=0, collate_fn=my_collate)

    
    elif mod == 'test1':
        #worker = worker_orignal['noise'].unique()[worker_idx]
        output_size = (224, 224)
        print("----worker number: %2d---- %s" %(worker_idx, worker))
        cross_data_path = '/home/user/use_trans/LIVE_WILD/image_labeled_by_score.csv'
        transformed_dataset_valid_1 = ImageRatingsDataset(csv_file=cross_data_path,
                                                        root_dir='/home/user/data/LIVEwild/ChallengeDB_release/Images',
                                                        transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        dataloader_train = 0
        dataloader_valid = DataLoader(transformed_dataset_valid_1, batch_size= b_s_v,
                                        shuffle=False, num_workers=0)

    elif mod == 'test2':
        output_size = (256, 256)
        cross_data_path = '/home/user/use_trans/LIVE_WILD/image_labeled_by_score.csv'
        transformed_dataset_valid_1 = ImageRatingsDataset(csv_file=cross_data_path,
                                                        root_dir='/home/user/data/LIVEwild/ChallengeDB_release/salient_images',
                                                        transform=transforms.Compose([#NIQEMax(patch_size=224, stride=112),
                                                                                      Rescale(output_size=(224, 224)),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        dataloader_train = 0
        dataloader_valid = DataLoader(transformed_dataset_valid_1, batch_size= b_s_v,
                                        shuffle=False, num_workers=0)


    else:
        output_size = (256, 256)
        cross_data_path = '/home/user/use_trans/LIVE_WILD/image_labeled_by_score.csv'
        transformed_dataset_valid_1 = ImageRatingsDataset(csv_file=cross_data_path,
                                                        root_dir='/home/user/data/LIVEwild/ChallengeDB_release/non_salient_images',
                                                        transform=transforms.Compose([#NIQEMin(patch_size=224, stride=112),
                                                                                      #RandomCrop(output_size=output_size),
                                                                                      Rescale(output_size=(224, 224)),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        dataloader_train = 0
        dataloader_valid = DataLoader(transformed_dataset_valid_1, batch_size= b_s_v,
                                        shuffle=False, num_workers=0)


    return dataloader_train, dataloader_valid


train_model()
