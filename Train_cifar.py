from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
from torch.utils.tensorboard import SummaryWriter
import pdb
import io
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from edl_losses import *
import warnings
from pathlib import Path
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--slr', '--s_learning_rate', default=0.02, type=float, help='initial learning rate for netS')
parser.add_argument('--dlr', '--d_learning_rate', default=0.02, type=float, help='initial learning rate for netD')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=1/3, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--on', default=0, type=float, help='open noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='../../data/cifar10/cifar-10-batches-py', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--noisy_dataset', default='cifar100', type=str)
parser.add_argument('--noise_dir', default='./noise', type=str)
parser.add_argument('--noise_data_dir', default='../../data/cifar100/cifar-100-python', type=str)
parser.add_argument('--inference', action='store_true')
parser.add_argument('--skip_warmup', action='store_true')
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--load_state_dict', default=None, type=str)
parser.add_argument('--warmup_epochs_S', default=30, type=int)
parser.add_argument('--warmup_epochs_D', default=10, type=int)
parser.add_argument('--plots_dir', default='plots/', type=str)
parser.add_argument('--gmmc', default=3, type=int)
parser.add_argument('--cudaid', default=0)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = device = torch.device('cuda:{}'.format(args.cudaid))

writer = SummaryWriter('runs/r={}_on={}'.format(args.r, args.on))

args.uid = '{}_{}'.format(args.dataset, args.noisy_dataset)

args.plots_dir = os.path.join(args.plots_dir, args.uid)
args.plots_dir = os.path.join(args.plots_dir, 'r={}_on={}'.format(args.r, args.on))
args.save_dicts_dir = os.path.join('saveDicts', args.uid)
args.checkpoint_dir = os.path.join('./checkpoint',args.uid)

Path(os.path.join(args.plots_dir, 'netS')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(args.plots_dir, 'netD')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(args.save_dicts_dir,)).mkdir(parents=True, exist_ok=True)
Path(os.path.join(args.checkpoint_dir,)).mkdir(parents=True, exist_ok=True)

# Training
def train_D(epoch,netD,optimizer,labeled_trainloader,unlabeled_trainloader):
    netD.train()

    global iter_net
    iter_idx = 1
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)  
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.to(device), inputs_x2.to(device), labels_x.to(device), w_x.to(device)
        inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)

        with torch.no_grad():
            # label guessing of unlabeled samples
            outputs_u1 = netD(inputs_u)
            outputs_u2 = netD(inputs_u2)
            
            pu = (torch.softmax(outputs_u1, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = netD(inputs_x)
            outputs_x2 = netD(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = netD(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.to(device)        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_net[iter_idx] += 1

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        writer.add_scalar('Train/Loss/{}/Labelled'.format('netD'), Lx.item(), iter_net[iter_idx])
        writer.add_scalar('Train/Loss/{}/Unlabelled'.format('netD'), Lu.item(), iter_net[iter_idx])
        sys.stdout.flush()

def train_S(epoch,netS,optimizer,dataloader):
    netS.train()

    global iter_net
    iter_idx = 0
    
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = netS(inputs)
        y = one_hot_embedding(labels.type(torch.LongTensor))
        y = y.to(device)
        loss = subjective_loss(outputs, y.float()).mean()
        loss.backward()  
        optimizer.step() 
        iter_net[iter_idx] += 1
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        writer.add_scalar('Train/Loss/All{}'.format('netS'), loss.item(), iter_net[iter_idx])
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader,model_name):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = net(inputs)
        if model_name == 'netS':
            warm_up_epochs = args.warmup_epochs_S
            y = one_hot_embedding(labels)
            y = y.to(device)
            loss = subjective_loss(outputs, y.float()).mean()
        else:          
            warm_up_epochs = args.warmup_epochs_D     
            loss = CEloss(outputs, labels)      
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, warm_up_epochs, batch_idx+1, num_iter, loss.item()))
        writer.add_scalar('Warmup/Loss/{}'.format(model_name), loss.item(), epoch * num_iter + batch_idx)
        sys.stdout.flush()
    losses = get_losses_on_all(net, model_name)

def test(epoch,net,test_loader,model_name):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()
    writer.add_scalar('Test/Accuracy/{}'.format(model_name), acc, epoch)

def get_losses_on_all(model, model_name):
    eval_loader = loader.run('eval_train')
    model.eval()
    losses = torch.zeros(50000)   
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device) 
            outputs = model(inputs)
            if model_name == 'netS':
                y = one_hot_embedding(targets)
                y = y.to(device)
                loss = subjective_loss(outputs, y.float())
            else:               
                loss = CE(outputs, targets)         
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]    
    losses = (losses-losses.min())/(losses.max()-losses.min())    # normalised losses for each image
    return losses

def refine_labels(model, probs, zero_open=False):
    probClean, probOpen, probClosed = probs[0], probs[1], probs[2]
    predClean = (probClean > probOpen) & (probClean > probClosed)      
    predClosed = (probClosed > probClean) & (probClosed > probOpen)
    predOpen = (probOpen > probClean) & (probOpen > probClosed)

    model.eval()
    targets = torch.zeros(50000)

    w_x = probClosed
    w_x = torch.from_numpy(np.expand_dims(w_x, axis=1)).to(device)
   
    eval_loader = loader.run('eval_train')
    with torch.no_grad():
        for batch_idx, (inputs, labels, index) in enumerate(eval_loader):
            inputs, labels = inputs.to(device), one_hot_embedding(labels).to(device) 
            outputs = model(inputs)             

            px = torch.softmax(outputs, dim=1).to(device)
            px = (1-w_x[index])*labels + (w_x[index])*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            refined = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            refined = refined.detach()       
            for b in range(inputs.size(0)):
                if zero_open:
                    if predOpen[index[b]]:
                            targets[index[b]] = -100
                    else:
                        targets[index[b]] = refined[b].argmax()
                else:
                    targets[index[b]] = refined[b].argmax()
    return targets

def fit_gmm_multiple_components(input_loss):
    gmm = GaussianMixture(n_components=20,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    components_open = []
    components_clean = []
    components_closed = []
    for n in range(gmm.n_components):
        if (gmm.means_[n] > .3) & (gmm.means_[n] < .7):
            components_open.extend([n])
        elif (gmm.means_[n] < .3):
            components_clean.extend([n])
        else:
            components_closed.extend([n])
    prob = gmm.predict_proba(input_loss)
    # transform this probability into a 3-component probability
    prob_clean = np.sum(prob[:,components_clean],axis=1)
    prob_closed = np.sum(prob[:,components_closed],axis=1)
    prob_open = np.sum(prob[:,components_open], axis=1)
    return prob_clean, prob_open, prob_closed

def eval_train(model,all_loss,epoch,model_name):    
    losses = get_losses_on_all(model, model_name)
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)

    strongClean = input_loss.argmin()
    strongClosed = input_loss.argmax()

    probClean, probOpen, probClosed = fit_gmm_multiple_components(input_loss)

    predClean = (probClean > probOpen) & (probClean > probClosed)      
    predClosed = (probClosed > probClean) & (probClosed > probOpen)
    predOpen = (probOpen > probClean) & (probOpen > probClosed)
    # guarantee that there is at least one sample in clean and closed
    predClean[strongClean] = True
    predClosed[strongClosed] = True
    return [probClean, probOpen, probClosed], [predClean, predOpen, predClosed], all_loss

def linear_rampup(current, rampup_length=16):
    current = np.clip((current) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

def printDataSplit(predClean, predOpen, predClosed):
    cleanIdx = np.where(predClean)[0]
    closedIdx = np.where(predClosed)[0]
    openIdx = np.where(predOpen)[0]    
    clean, open_noise, closed_noise = loader.get_noise()
    stats_log.write('Dividing dataset...\n')
    if len(clean) != 0:
        stats_log.write('Clean - clean:{:.2f}, closed:{:.2f}, open;{:.2f}\n'.format(len(set(cleanIdx).intersection(clean))/len(clean), len(set(closedIdx).intersection(clean))/len(clean), len(set(openIdx).intersection(clean))/len(clean)))
    if len(closed_noise) != 0:
        stats_log.write('Closed - clean:{:.2f}, closed:{:.2f}, open;{:.2f}\n'.format(len(set(cleanIdx).intersection(closed_noise))/len(closed_noise), len(set(closedIdx).intersection(closed_noise))/len(closed_noise), len(set(openIdx).intersection(closed_noise))/len(closed_noise)))
    if len(open_noise) != 0:
        stats_log.write('Open - clean:{:.2f}, closed:{:.2f}, open;{:.2f}\n\n'.format(len(set(cleanIdx).intersection(open_noise))/len(open_noise), len(set(closedIdx).intersection(open_noise))/len(open_noise), len(set(openIdx).intersection(open_noise))/len(open_noise)))
    stats_log.flush()

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.to(device)
    return model

def plotHistogram(data, predictions=None, log=False, model_name='', phase='', title=''):
    clean, open_noise, closed_noise = loader.get_noise()
    fig = plt.figure()
    if predictions is not None:
        plt.subplot(121)
    plt.hist(data[clean], bins=300, alpha=0.5, color='green')
    plt.hist(data[closed_noise], bins=300, alpha=0.5, color='blue')
    plt.hist(data[open_noise], bins=300, alpha=0.5, color='red')
    # plt.legend(loc='upper right')
    if predictions is not None:
        plt.subplot(122)
        plt.hist(data[predictions[0]], bins=300, alpha=0.5, color='green', label='Predicted clean set')
        plt.hist(data[predictions[2]], bins=300, alpha=0.5, color='blue', label='Predicted closed set')
        plt.hist(data[predictions[1]], bins=300, alpha=0.5, color='red', label='Predicted open set')
        plt.legend(loc='upper right')
    if log:
        print('\nlogging histogram...')
        plt.savefig(os.path.join(args.plots_dir, '{}/{}_{}'.format(model_name, phase, title)), format='png')
    else:
        plt.show()
    plt.close()

def get_logits(model):
    eval_loader = loader.run('eval_train')
    model.eval()
    logits = np.zeros((50000, 10))   
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device) 
            outputs = model(inputs)
            logits[index] = outputs.cpu()
    return logits

def runExperiment():
    print('| Building net')
    netS = create_model()
    netD = create_model()
    if args.load_state_dict is not None:
        print('Loading saved state dict from {}'.format(args.load_state_dict))
        checkpoint = torch.load(args.load_state_dict)
        netS.load_state_dict(checkpoint['netS_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
    cudnn.benchmark = True
    optimizer1 = optim.SGD(netS.parameters(), lr=args.slr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(netD.parameters(), lr=args.dlr, momentum=0.9, weight_decay=5e-4)

    if args.noise_mode=='asym':
        conf_penalty = NegEntropy()

    all_loss = [[],[]] # save the history of losses from two networks
    test_loader = loader.run('test')

    if not args.skip_warmup:
        warmup_trainloader = loader.run('warmup')
        print('Warmup netS')
        for epoch in range(args.warmup_epochs_S):
            warmup(epoch,netS,optimizer1,warmup_trainloader,'netS')
            if epoch % 3 == 0:
                plotHistogram(get_losses_on_all(netS, 'netS'), log=True, model_name='netS', phase='warmup', title='epoch={}'.format(epoch))
            test(epoch,netS,test_loader,'netS')
        print('\nWarmup netD')
        for epoch in range(args.warmup_epochs_D):    
            warmup(epoch,netD,optimizer2,warmup_trainloader,'netD')
            if epoch % 3 == 0:
                plotHistogram(get_losses_on_all(netD, 'netD'), log=True, model_name='netD', phase='warmup', title='epoch={}'.format(epoch))
            test(epoch,netD,test_loader,'netD') 
        print('\nSaving warmup state dict...')
        torch.save({
        'netS_state_dict': netS.state_dict(),
        'netD_state_dict': netD.state_dict(),
        }, os.path.join(args.save_dicts_dir, 'warmup_%.1f_%0.2f.json'%(args.r,args.on)))
    else:
        test(args.start_epoch-1,netS,test_loader,'netS')
        test(args.start_epoch-1,netD,test_loader,'netD')

    for epoch in range(args.start_epoch, args.num_epochs+1):
        slr=args.slr
        dlr=args.dlr
        if epoch > 100:
            dlr /= 10      
        for param_group in optimizer1.param_groups:
            param_group['lr'] = slr       
        for param_group in optimizer2.param_groups:
            param_group['lr'] = dlr          
        clean_loader = loader.run('clean')
        eval_loader = loader.run('eval_train')   
        #train_clean_accuracy(epoch, netS, netD, clean_loader)
    
        probs, preds, all_loss[0] = eval_train(netS,all_loss[0], epoch, 'netS')         

        probClean, probOpen, probClosed = probs[0], probs[1], probs[2]    
        predClean, predOpen, predClosed = preds[0], preds[1], preds[2]      

        printDataSplit(predClean, predOpen, predClosed)

        if epoch % 3 == 0:
            plotHistogram(get_losses_on_all(netS, 'netS'), [predClean, predOpen, predClosed], log=True, model_name='netS', phase='train', title='epoch={}'.format(epoch))
            plotHistogram(get_losses_on_all(netD, 'netD'), [predClean, predOpen, predClosed], log=True, model_name='netD', phase='train', title='epoch={}'.format(epoch))

        print('\nTrain netD')
        labeled_trainloader, unlabeled_trainloader = loader.run('trainD', predClean, predClosed, probClean) # divide
        train_D(epoch,netD,optimizer2,labeled_trainloader, unlabeled_trainloader) # train netD  
        
        #refine labels using netD
        targets = refine_labels(netD, [probClean, probOpen, probClosed])

        print('\nTrain netS')
        trainloader = loader.run('trainS', targets=targets)
        train_S(epoch,netS,optimizer1,trainloader)

        test(epoch,netS,test_loader,'netS')
        test(epoch,netD,test_loader, 'netD')

        if epoch % 10 == 0:
            torch.save({
                'netS_state_dict': netS.state_dict(),
                'netD_state_dict': netD.state_dict(),
                }, os.path.join(args.save_dicts_dir, 'train_%.1f_%0.2f_%d.json'%(args.r,args.on,epoch)))

def train_clean_accuracy(epoch,netS,netD,clean_loader):
    netS.eval()
    netD.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(clean_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1 = netS(inputs)
            outputs2 = netD(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)                          
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Clean Set Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()
    writer.add_scalar('Clean/Accuracy', acc, epoch)

def infer():
    print('Loading saved state dict from {}'.format(args.load_state_dict))
    checkpoint = torch.load(args.load_state_dict)
    netS = create_model()
    netD = create_model()
    netS.load_state_dict(checkpoint['netS_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    losses = get_losses_on_all(netS, 'netS')
    plotHistogram(losses)

stats_log=open('./%s/%.1f_%.2f'%(args.checkpoint_dir,args.r,args.on)+'_stats.txt','w') 
test_log=open('./%s/%.1f_%.2f'%(args.checkpoint_dir,args.r,args.on)+'_acc.txt','w')

loader = dataloader.cifar_dataloader(args.dataset, args.noisy_dataset, r=args.r, on=args.on, noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path, noise_data_dir=args.noise_data_dir, log=stats_log,noise_file='%s/%.1f_%0.2f_%s.json'%(args.noise_dir,args.r,args.on,args.noisy_dataset))

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
subjective_loss = edl_mse_loss
criterion = SemiLoss()
iter_net = [0, 0]

if args.inference is False:
    runExperiment()
else:
    infer()