from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import scipy.io as sio
import torch
from torchnet.meter import AUCMeter
import pdb

            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, noisy_dataset, r, on, noise_mode, root_dir, noise_data_dir, transform, mode, noise_file='', pred=[], probability=[], log='', targets=None): 
        
        self.r = r # total noise ratio
        self.on = on # proportion of open noise
        self.transform = transform
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        self.open_noise = None
        self.closed_noise = None
     
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']
       
        elif self.mode=='clean':
            if not os.path.exists(noise_file):
                print('Noise not defined')
                return
            
            if self.open_noise is None or self.closed_noise is not None:
                noise = json.load(open(noise_file,"r"))
                noise_labels = noise['noise_labels']
                self.open_noise = noise['open_noise']
                self.closed_noise = noise['closed_noise']

            train_data=[]
            train_label=[]
            noise_data=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            open_noise = [item[0] for item in self.open_noise]
            clean_indices = list(set(range(50000)) - set(open_noise) - set(self.closed_noise))
            self.clean_data = train_data[clean_indices]
            self.clean_label = np.asarray(train_label)[clean_indices]

        else:    
            train_data=[]
            train_label=[]
            noise_data=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            if noisy_dataset == 'imagenet32':
                noise_data = None
            else:
                noise_data = unpickle('%s/train'%noise_data_dir)['data'].reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))
        
            if os.path.exists(noise_file):
                noise = json.load(open(noise_file,"r"))
                noise_labels = noise['noise_labels']
                self.open_noise = noise['open_noise']
                self.closed_noise = noise['closed_noise']
                for cleanIdx, noisyIdx in noise['open_noise']:
                    if noisy_dataset == 'imagenet32':
                        train_data[cleanIdx] = np.asarray(Image.open('{}/{}.png'.format(noise_data_dir, str(noisyIdx+1).zfill(7)))).reshape((32,32,3))
                    else:
                        train_data[cleanIdx] = noise_data[noisyIdx]
            else:
                #inject noise   
                noise_labels = []                       # all labels (some noisy, some clean)
                idx = list(range(50000))                # indices of cifar dataset
                random.shuffle(idx)                 
                num_total_noise = int(self.r*50000)     # total amount of noise
                num_open_noise = int(self.on*num_total_noise)     # total amount of noisy/openset images
                if noisy_dataset == 'imagenet32':       # indices of openset source images
                    target_noise_idx = list(range(1281149))
                else:
                    target_noise_idx = list(range(50000))
                random.shuffle(target_noise_idx)  
                self.open_noise = list(zip(idx[:num_open_noise], target_noise_idx[:num_open_noise]))  # clean sample -> openset sample mapping
                self.closed_noise = idx[num_open_noise:num_total_noise]      # closed set noise indices
                # populate noise_labels
                for i in range(50000):
                    if i in self.closed_noise:
                        if noise_mode=='sym':
                            if dataset=='cifar10': 
                                noiselabel = random.randint(0,9)
                            elif dataset=='cifar100':    
                                noiselabel = random.randint(0,99)
                            noise_labels.append(noiselabel)
                        elif noise_mode=='asym':   
                            noiselabel = self.transition[train_label[i]]
                            noise_labels.append(noiselabel)               
                    else:
                        noise_labels.append(train_label[i])
                # populate openset noise images
                for cleanIdx, noisyIdx in self.open_noise:
                    if noisy_dataset == 'imagenet32':
                        train_data[cleanIdx] = np.asarray(Image.open('{}/{}.png'.format(noise_data_dir, str(noisyIdx+1).zfill(7)))).reshape((32,32,3))
                    else:
                        train_data[cleanIdx] = noise_data[noisyIdx]
                # write noise to a file, to re-use
                noise = {'noise_labels': noise_labels, 'open_noise': self.open_noise, 'closed_noise': self.closed_noise}
                print("save noise to %s ..."%noise_file)       
                json.dump(noise,open(noise_file,"w"))       
            
            if self.mode == 'all':
                self.train_data = train_data
                if targets is None:
                    self.noise_labels = noise_labels
                else:
                    self.noise_labels = targets
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_labels)==np.array(train_label))                                                    
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)
                    # note: If all the labels are clean, the following will return NaN       
                    auc,_,_ = auc_meter.value()                     
                    
                elif self.mode == "unlabeled":
                    pred_idx = pred.nonzero()[0]                                               
                
                self.train_data = train_data[pred_idx]
                self.noise_labels = [noise_labels[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_labels)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_labels[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)               
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
        elif self.mode=='clean':
            img, target = self.clean_data[index], self.clean_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode=='clean':
            return len(self.clean_data)
        elif self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_noise(self):
        return (self.open_noise, self.closed_noise)       
        
        
class cifar_dataloader():  
    def __init__(self, dataset, noisy_dataset, r, on, noise_mode, batch_size, num_workers, root_dir, noise_data_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.on = on
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_data_dir = noise_data_dir
        self.log = log
        self.noise_file = noise_file
        self.open_noise = None
        self.closed_noise = None
        self.noisy_dataset = noisy_dataset

        if self.dataset=='cifar10':
            # todo: normalise the noise dataset properly
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
    def run(self,mode,predClean=[], predClosed=[], probClean=[], targets=None):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noisy_dataset=self.noisy_dataset, noise_mode=self.noise_mode, r=self.r, on=self.on, root_dir=self.root_dir, noise_data_dir=self.noise_data_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)  
            self.open_noise, self.closed_noise = all_dataset.get_noise()           
            return trainloader
                                     
        elif mode=='trainD':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noisy_dataset=self.noisy_dataset, noise_mode=self.noise_mode, r=self.r, on=self.on, root_dir=self.root_dir, noise_data_dir=self.noise_data_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=predClean, probability=probClean,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noisy_dataset=self.noisy_dataset, noise_mode=self.noise_mode, r=self.r, on=self.on, root_dir=self.root_dir, noise_data_dir=self.noise_data_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=predClosed)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)    

            self.open_noise, self.closed_noise = labeled_dataset.get_noise()  
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='trainS':
            all_dataset = cifar_dataset(dataset=self.dataset, noisy_dataset=self.noisy_dataset, noise_mode=self.noise_mode, r=self.r, on=self.on, root_dir=self.root_dir, noise_data_dir=self.noise_data_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file, targets=targets)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)           
            return trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noisy_dataset=self.noisy_dataset, noise_mode=self.noise_mode, r=self.r, on=self.on, root_dir=self.root_dir, noise_data_dir=self.noise_data_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='clean':
            clean_dataset = cifar_dataset(dataset=self.dataset, noisy_dataset=self.noisy_dataset, noise_mode=self.noise_mode, r=self.r, on=self.on, root_dir=self.root_dir, noise_data_dir=self.noise_data_dir, transform=self.transform_test, mode='clean', noise_file=self.noise_file)      
            clean_loader = DataLoader(
                dataset=clean_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return clean_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noisy_dataset=self.noisy_dataset, noise_mode=self.noise_mode, r=self.r, on=self.on, root_dir=self.root_dir, noise_data_dir=self.noise_data_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            self.open_noise, self.closed_noise = eval_dataset.get_noise()         
            return eval_loader        
    
    def get_noise(self):
        open_noise = [item[0] for item in self.open_noise]
        clean = list(set(range(50000)) - set(open_noise) - set(self.closed_noise))
        return (clean, open_noise, self.closed_noise)
