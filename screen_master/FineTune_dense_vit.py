from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np
import cv2

from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torch.nn.parallel import DataParallel

from PIL import Image
import time
import math
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import timm
import one_load
import vit

import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr, pearsonr
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

torch.backends.cudnn.benchmark = True
ResultSave_path='record_freeze_vit.txt'

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


class ImageRatingsDataset2(Dataset):
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
            im = Image.open(img_name).convert('L')
            if im.mode == 'P':
                im = im.convert('L')

            img_dct = cv2.dct(np.array(im, np.float32))  #get dct image

            image = np.asarray(img_dct)
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
        self.fc3 = nn.Linear(512, 1)
        self.bn3 = nn.BatchNorm1d(1)              #add norm
        #self.sig = nn.Sigmoid()

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
        x = self.fc1(x)
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        #print(out.shape)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.bn3(out)
        #out = self.sig(out)
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
    correct = 0
    total = 0
    count = 0
    with torch.no_grad():
        #count += 1
        cum_loss = 0
        for data1 in dataloader_valid1:
            count += 1
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

def finetune_model():
    epochs = 50
    srocc_l = []
    plcc_l = []
    epoch_record = []
    
    print('=============Saving Finetuned Prior Model===========')
    data_dir = os.path.join('/home/user/designrice/')
    images = pd.read_csv(os.path.join(data_dir, 'data3.csv'), sep=',')
    images_fold = "/home/user/designrice/"
    if not os.path.exists(images_fold):
        os.makedirs(images_fold)
    for i in range(10):
        best_predicted = 0
        with open(ResultSave_path, 'a') as f1:  # 设置文件对象data.txt
            print(i,file=f1)

        print('\n')
        print('--------- The %2d rank trian-test (24epochs) ----------' % i )
        images_train, images_test = train_test_split(images, train_size = 0.8)

        train_path = images_fold + "train_image" + ".csv"
        test_path = images_fold + "test_image" + ".csv"
        images_train.to_csv(train_path, sep=',', index=False)
        images_test.to_csv(test_path, sep=',', index=False)
        
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
        
        model = torch.load('model_IQA/TID2013_Kadid_Meta.pt')
        '''
        for name, param in model.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}")
        #model.load_state_dict(torch.load('model_IQA/TID2013_KADID10K_IQA_Meta_densenet_newload.pt'))
        #model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        '''

        for m in model.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')
             
        '''
        for param in model.Vit.parameters():
            param.requires_grad = False
        '''
        
        criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-4,  weight_decay=0)
        model.cuda()      
        #model = DataParallel(model)
        
        best_predicted = 0
        for epoch in range(epochs):
            optimizer = exp_lr_scheduler(optimizer, epoch)
            count = 0

            if epoch == 0:
                dataloader_valid1 = load_data('train1')

                model.eval()

                sp = computeSpearman(dataloader_valid1, model)[0]
                if sp > best_predicted:
                    best_predicted = sp
                print('no train srocc {:4f}'.format(sp))

            # Iterate over data.
            #print('############# train phase epoch %2d ###############' % epoch)
            dataloader_train1 = load_data('train1')

            model.train()  # Set model to training mode
            running_loss = 0.0
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

                #print(labels1.shape)
                
                optimizer.zero_grad()
                outputs = model(inputs1)
                #print(outputs.shape)
                #labels1 = labels1.squeeze()
                loss = criterion(outputs, labels1)
                loss.backward()
                optimizer.step()
                
                #print('t  e  s  t %.8f' %loss.item())
                try:
                    running_loss += loss.item()

                except:
                    print('unexpected error, could not calculate loss or do a sum.')

                count += 1

            epoch_loss = running_loss / count
            epoch_record.append(epoch_loss)
            print(' The %2d epoch : current loss = %.8f ' % (epoch,epoch_loss))

            #print('############# test phase epoch %2d ###############' % epoch)
            dataloader_valid1 = load_data('test1')

            model.eval()
            
            sp, pl = computeSpearman(dataloader_valid1, model)
            
            if sp > best_predicted:
                best_predicted = sp
                print('=====Prior model saved===predicted:%f========'%sp)
                best_model = copy.deepcopy(model)
                torch.save(best_model.cuda(),'model_IQA/juhua.pt')
            
            print('Validation Results - Epoch: {:2d}, , predicted_plcc: {:4f}, predicted_srcc: {:4f}, best_srcc: {:4f}, '
                  .format(epoch, pl, sp, best_predicted))


    '''
    epoch_count = 0
    f = open('loss_record.txt','w')
    for line in epoch_record:
        epoch_record += 1
        f.write('epoch' + epoch_count + line + '\n')
        if epoch_record == 100:
            epoch_record = 0
    f.save()
    f.close()
    '''
    # ind = 'Results/LIVEWILD'
    # file = pd.DataFrame(columns=[ind], data=srocc_l)
    # file.to_csv(ind+'.csv')
    # print('average srocc {:4f}'.format(np.mean(srocc_l)))

def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=10):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.8**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_data(mod = 'train1'):

    meta_num = 24
    data_dir = os.path.join('/home/user/designrice/')
    train_path = os.path.join(data_dir,  'train_image.csv')
    test_path = os.path.join(data_dir,  'test_image.csv')

    output_size = (224, 224)
                                               
    transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir='/home/user/data/1/',
                                                    transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                  RandomHorizontalFlip(0.5),
                                                                                  RandomCrop(output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    
    transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir='/home/user/data/1/',
                                                    transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    
    bsize = meta_num

    if mod == 'train1':
        dataloader = DataLoader(transformed_dataset_train, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate, drop_last=True)
    
    else:
        dataloader = DataLoader(transformed_dataset_valid, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    

    return dataloader

finetune_model()
