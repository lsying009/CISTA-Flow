
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
print(os.environ["CUDA_VISIBLE_DEVICES"])

import torch
import torch.utils.data as data
from torch import optim, nn
from tensorboardX import SummaryWriter
import numpy as np
import argparse
import random

from utils.configs import set_configs
from data_readers.train_data_loaders import TrainfusedEventData
from e2v.e2v_model import *
from utils.evaluate import PerceptualLoss
from pytorch_msssim import SSIM
from utils.data_io import show_whole_img

from loss import FlowReconLoss #FlowL1LossDict
from utils.flow_utils import FrameWarp

from DCEIFlow.utils.utils import setup_seed


class Train:
    '''Train CISTA-Flow after getting pretrained CISTA (GT Flow) and DCEIFlow/ERAFT (GT I)'''
    def __init__(self, cfgs, device):
        # self.image_dim = cfgs.image_dim
        self.device = device
        if cfgs.model_name:
            self.model_name =  '{}_{}_b{}_d{}_c{}'.format(cfgs.model_name, cfgs.model_mode, \
                    cfgs.num_bins, cfgs.depth, cfgs.base_channels)
        else:
            self.model_name =  '{}_b{}_d{}_c{}'.format(cfgs.model_mode, \
                    cfgs.num_bins, cfgs.depth, cfgs.base_channels)
        self.path_to_model = os.path.join(cfgs.path_to_model, self.model_name)
        if not os.path.exists(self.path_to_model):
            os.makedirs(self.path_to_model)
            
        # Loss
        self.frame_warp = FrameWarp(mode=cfgs.warp_mode)
        self.loss_fn = FlowReconLoss(cfgs.image_dim, self.frame_warp,  ds=cfgs.ds, is_bi=False, lpips_net='vgg').to(self.device) #self.device
    
        self.model_mode = cfgs.model_mode
        if self.model_mode == 'cista-eiflow':
            self.model = DCEIFlowCistaNet(cfgs)
            if cfgs.distributed:
                self.model = DCEIFlowCistaNet2GPU(cfgs)
        elif self.model_mode == 'cista-eraft':
            self.model = ERAFTCistaNet(cfgs)
        else:
            assert self.model_mode in ['cista-eiflow', 'cista-eraft']

        
        if cfgs.load_epoch_for_train:
            checkpoint = torch.load(os.path.join(self.path_to_model, '{}_{}.pth.tar'\
                                .format(self.model_name, cfgs.load_epoch_for_train)), map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'], strict=True) #True
        elif cfgs.path_to_e2vflow: # if having pretrained CISTA-Flow network
            checkpoint = torch.load(cfgs.path_to_e2vflow, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('Load path_to_e2vflow: {}'.format(cfgs.path_to_e2vflow))
        else:
            # load pretrained reconstruction network CISTA (GT Flow)
            if cfgs.path_to_e2v:
                checkpoint = torch.load(cfgs.path_to_e2v, map_location='cpu')
                self.model.cista_net.load_state_dict(checkpoint['state_dict'], strict=True)
                print('Load path_to_e2v: {}'.format(cfgs.path_to_e2v))
            else:
                assert cfgs.path_to_e2v, "Should load pretrained CISTA (GT Flow)"
            # load pretrained flow network
            if cfgs.path_to_flownet:
                checkpoint = torch.load(cfgs.path_to_flownet, map_location='cpu')
                self.model.event_flownet.load_state_dict(checkpoint['state_dict'], strict=True)
                print('Load path_to_flownet: {}'.format(cfgs.path_to_flownet))
            else:
                assert cfgs.path_to_flownet, "Should load pretrained DCEIFlow/ERAFT (GT I)"
            
        if not cfgs.distributed:
            self.model = self.model.to(device)
        self.model.train()
        print(self.model)
        
        
        # Load training data
        path_to_train_data = cfgs.path_to_train_data
        train_data = TrainfusedEventData(os.path.join(path_to_train_data, 'train_e2v_estflow.txt'), cfgs)
        self.train_loader = data.DataLoader(train_data,batch_size=cfgs.batch_size, shuffle=cfgs.shuffle, num_workers=4)  # num_workers=4----------- 
        
        lr = cfgs.lr*(0.9**np.floor(cfgs.load_epoch_for_train/10.)) 
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        
        
        # Save training results
        self.is_SummaryWriter = cfgs.is_SummaryWriter
        if self.is_SummaryWriter:
            self.writer = SummaryWriter('./summary/{}'\
            .format(self.model_name)) 
        
        

    def run_train(self, cfgs):
        '''
            1. [0, flow_epoch], train DCEIFlow/ERAFT (Rec I), fix CISTA (GT Flow)
            2. [flow_epoch, flow_epoch + rec_epoch], train CISTA (Pred Flow), fix DCEIFlow/ERAFT (Rec I)
            3. [flow_epoch + rec_epoch, epoch], train CISTA-Flow without GT data iteratively
        '''

        for epoch in range(cfgs.load_epoch_for_train, cfgs.epochs):
            lr = self.scheduler.get_last_lr()[0]
            
            if epoch <cfgs.flow_epoch:
                self.model.fix_params(net_name='rec')
                train_rec = False
            elif epoch >=cfgs.flow_epoch and epoch < cfgs.flow_epoch + cfgs.rec_epoch:
                self.model.fix_params(net_name='flow')
                train_rec = True
            else:
                self.optimizer.param_groups[0]['lr'] = 3e-5
                if (epoch-cfgs.flow_epoch - cfgs.rec_epoch)%4 >=2:
                    self.model.fix_params(net_name='flow')
                    train_rec = True
                else:
                    self.model.fix_params(net_name='rec')
                    train_rec = False
            print('lr:', self.optimizer.param_groups[0]['lr'])
            print('train_rec: ', train_rec)    
            
            self.train_epoch(epoch, cfgs, train_rec, 'cuda:0' if cfgs.distributed else self.device)
 
            self.scheduler.step()
            
            if epoch == 0 or (epoch+1)==cfgs.flow_epoch+cfgs.rec_epoch or ((epoch+1)>=cfgs.flow_epoch + cfgs.rec_epoch and (epoch+1-cfgs.flow_epoch-cfgs.rec_epoch)%2 == 0) or (epoch+1) % 10 == 0: # + cfgs.rec_epoch or (epoch+1) % 10 == 0:
                torch.save({'epoch': epoch, 'state_dict': self.model.state_dict()}, 
                            os.path.join(self.path_to_model, '{}_{}.pth.tar'\
                                .format(self.model_name, epoch+1)))    


    def run_train_distributed(self, cfgs):
        '''
            1. [0, flow_epoch], train DCEIFlow/ERAFT (Rec I), fix CISTA (GT Flow)
            2. [flow_epoch, flow_epoch + rec_epoch], train CISTA (Pred Flow), fix DCEIFlow/ERAFT (Rec I)
            3. [flow_epoch + rec_epoch, epoch], train CISTA-Flow without GT data iteratively
        '''

        for epoch in range(cfgs.load_epoch_for_train, cfgs.epochs):
            lr = self.scheduler.get_last_lr()[0]
            
            if epoch <cfgs.flow_epoch:
                self.model.module.fix_params(net_name='rec')
                train_rec = False
            elif epoch >=cfgs.flow_epoch and epoch < cfgs.flow_epoch + cfgs.rec_epoch:
                self.model.module.fix_params(net_name='flow')
                train_rec = True
            else:
                self.optimizer.param_groups[0]['lr'] = 3e-5
                if (epoch-cfgs.flow_epoch - cfgs.rec_epoch)%4 >=2:
                    self.model.module.fix_params(net_name='flow')
                    train_rec = True
                else:
                    self.model.module.fix_params(net_name='rec')
                    train_rec = False
            print('lr:', self.optimizer.param_groups[0]['lr'])
            print('train_rec: ', train_rec)    
            
            self.train_epoch(epoch, cfgs, train_rec, 'cuda:0' if cfgs.distributed else self.device)
 
            self.scheduler.step()
            
            if epoch == 0 or (epoch+1)==cfgs.flow_epoch+cfgs.rec_epoch or ((epoch+1)>=cfgs.flow_epoch + cfgs.rec_epoch and (epoch+1-cfgs.flow_epoch-cfgs.rec_epoch)%2 == 0) or (epoch+1) % 10 == 0: # + cfgs.rec_epoch or (epoch+1) % 10 == 0:
                torch.save({'epoch': epoch, 'state_dict': self.model.module.state_dict()}, 
                            os.path.join(self.path_to_model, '{}_{}.pth.tar'\
                                .format(self.model_name, epoch+1)))    


    def train_epoch(self, epoch, cfgs, train_rec=False, device='cuda:0'):
        torch.cuda.empty_cache()

        batch_num =len(self.train_loader)
        loss = 0
        states = None
        output = None
        
        for batch_idx, seq_data in enumerate(self.train_loader):
            loss = 0
            cur_gt = dict([])
            for s in range(len(seq_data)): 
                cur_data = {key: value.to(device) for key, value in seq_data[s][0].items()}
                cur_target = {key: value.to(device) for key, value in seq_data[s][1].items()}
                
                if s == 0:
                    cur_data['rec_img0'] = torch.zeros_like(cur_target['gt_img1'])
                    states = None 
                else:
                    cur_data['rec_img0'] = output.clone()
                

                cur_gt['gt_img1'] = cur_target['gt_img1'].clone()

                # when training DCEIFlow (Rec I), provide gt_flow for CISTA (GT Flow)
                if epoch < cfgs.flow_epoch: 
                    cur_gt['gt_flow'] = cur_target['gt_flow'].clone()
                
                output, batch_flow, states = self.model(cur_data, states, cur_gt)   

                if train_rec:
                    loss_mode = 'rec'
                    is_loss_consis = True if s >=2 else False
                else:
                    loss_mode = 'flow'
                    is_loss_consis = False
                    if epoch >= cfgs.flow_epoch+cfgs.rec_epoch:
                        loss_mode = 'both'
                
                # Print memory usage
                # print(s, f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                # print(s, f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
                
                loss += self.loss_fn(output, cur_data['rec_img0'], batch_flow, cur_target, loss_mode, is_loss_consis=is_loss_consis)
                

            if self.is_SummaryWriter:
                self.writer.add_scalar('loss', loss, batch_num*epoch+batch_idx)


            self.optimizer.zero_grad()
            loss.backward(retain_graph=False) 
            self.optimizer.step() 
            

            if batch_idx%50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(\
                    epoch+1, batch_idx*self.train_loader.batch_size, len(self.train_loader.dataset),\
                    100.*batch_idx/len(self.train_loader), loss.data)) # .data.cpu().numpy()


if __name__ == '__main__':
    # seed = 1234
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    
    setup_seed(1234)

    
    ## config parameters
    parser = argparse.ArgumentParser(
        description='Training options')
    set_configs(parser)
    cfgs = parser.parse_args()
    cfgs.shuffle = True

    if cfgs.distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    model_train = Train(cfgs, device)
    if cfgs.distributed:
        model_train.run_train_distributed(cfgs)
    else:
        model_train.run_train(cfgs)
   

