import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import csv
import os

from utils.image_process import normalize_image
from data_readers.video_readers import ImageReader
from e2v.e2v_model import * 

from utils.configs import set_configs
from utils.data_io import * #ImageWriter, EvalWriter, FlowWriter, EventWriter, show_whole_img, show_flow
from utils.event_process import event_preprocess, events_to_voxel_grid, events_to_voxel_grid_pol
from data_readers.event_readers import FixedSizeEventReader, SingleEventReaderNpz

# from superslomo.model import backWarp
from utils.flow_utils import FrameWarp


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=True).type(dtype).to(x.device)
    x = F.grid_sample(x, grid)
    return x



class Reconstructor(nn.Module):
    def __init__(self, cfgs, device, data_folder=None):
        super(Reconstructor, self).__init__()
        self.image_dim = cfgs.image_dim
        self.reader_type = cfgs.reader_type
        self.model_mode = cfgs.model_mode
        self.device = device
        self.num_load_frames = cfgs.test_img_num
        self.test_data_name = cfgs.test_data_name
        self.limit_num_events = cfgs.num_events
        self.test_data_mode = cfgs.test_data_mode 
        self.warp_mode = cfgs.warp_mode
        self.display_test = cfgs.display_test
        
        self.num_bins = cfgs.num_bins
        self.k_shift = cfgs.k_shift
        self.n_event_skip = cfgs.n_event_skip
        # self.len_skip_frames = cfgs.len_skip_frames
        self.dvs = cfgs.dvs
        
        
        self.path_to_sequences = []
        for folder_name in os.listdir(cfgs.path_to_test_data):
            if os.path.isdir(os.path.join(cfgs.path_to_test_data, folder_name)) or folder_name.split('/')[-1].split('.')[-1] == 'zip':
                self.path_to_sequences.append(os.path.join(cfgs.path_to_test_data, folder_name))

        self.path_to_sequences.sort()
        print(self.path_to_sequences)

        # initialize reconstruction network        
        if self.model_mode == 'cista-eiflow':
            self.model = DCEIFlowCistaNet(cfgs)
        elif self.model_mode == 'cista-eraft':
            self.model = ERAFTCistaNet(cfgs)
        else:
            assert self.model_mode in ['cista-eiflow', 'cista-eraft']
        
        if cfgs.path_to_e2v:
            checkpoint = torch.load(cfgs.path_to_e2v, map_location='cuda:0')
            self.model.cista_net.load_state_dict(checkpoint['state_dict'], strict=True)
            self.model_name = self.model_mode
        else:
            # Load pretrained model
            if cfgs.load_epoch_for_test:
                self.model_name = cfgs.path_to_test_model.split('/')[-2]
                cfgs.path_to_test_model = cfgs.path_to_test_model + '{}_{}.pth.tar'.format(self.model_name, cfgs.load_epoch_for_test)
                self.model_name = self.model_name + '/{}'.format(cfgs.load_epoch_for_test)
                
            else:
                self.model_name = os.path.splitext(cfgs.path_to_test_model.split('/')[-1])[0]
            checkpoint = torch.load(cfgs.path_to_test_model, map_location='cuda:0')
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)

        self.model.to(device)
        self.model.eval()

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('\n Number of parameters in {}: {:d}'.format(self.model_mode, params))

        self.frame_warp = FrameWarp(mode=cfgs.warp_mode)
            
    def forward(self):
        with torch.no_grad():
            for seq_id, path_to_sequence_folder in enumerate(self.path_to_sequences):
                dataset_name=path_to_sequence_folder.split('/')[-1].split('.')[0]
                print(dataset_name)
                # print(path_to_sequence_folder)
                if self.test_data_name is not None and dataset_name != self.test_data_name:
                    continue

                states = None
                prev_image = None
                flow_states = None
                # prev_events = None
                if os.path.isdir(path_to_sequence_folder):
                    path_to_events = []
                    for root, dirs, files in os.walk(path_to_sequence_folder):
                        for file_name in files:
                            if (file_name.split('.')[-1] in ['npz'] or file_name in ['events.txt', 'events.zip', 'events.csv']):
                                path_to_events.append(os.path.join(root, file_name))
                    path_to_events.sort()
                    event_window_iterator = SingleEventReaderNpz(path_to_events)
                else:
                    event_window_iterator = FixedSizeEventReader(path_to_sequence_folder, self.limit_num_events, self.k_shift, self.n_event_skip)
                image_writer = ImageWriter(cfgs, self.model_name, dataset_name)
                flow_writer = FlowWriter(cfgs, self.model_name, dataset_name)
                event_writer = EventWriter(cfgs, self.model_name, dataset_name)

                frame_idx = 0
                for event_window in event_window_iterator:
                    if frame_idx > self.num_load_frames:
                        break
                    
                    if prev_image is None:
                        prev_image = torch.zeros([1, 1, self.image_dim[0], self.image_dim[1]], dtype=torch.float32, device=self.device)


                    event_window = events_to_voxel_grid(event_window, 
                                                num_bins=self.num_bins,
                                                width=self.image_dim[1],
                                                height=self.image_dim[0])
                    
                    event_window = event_preprocess(event_window, filter_hot_pixel=True)
                    evs = torch.unsqueeze(torch.from_numpy(event_window), axis=0).to(self.device)
                    if self.dvs == 'samsung':
                        evs = rot_img(evs, np.pi, dtype=evs.dtype)
                    
                    input_data = dict(event_voxel = evs, 
                                      rec_img0=prev_image)
                    
                    if self.model_mode in ['cista-eiflow']:
                        pred_image, batch_flow, states = self.model(input_data, states) 
                    elif self.model_mode in ['cista-eraft']:
                        if frame_idx == 0:
                            evs_old = torch.zeros_like(evs)
                        input_data['event_voxel_old'] = evs_old
                        pred_image, batch_flow, states = self.model(input_data, states)
                        evs_old = evs.clone()  

                    prev_image = pred_image.clone()
                    # if cfgs.display_test:
                    #     # show_whole_img(evs, init_flow, pred_flow) #torch.from_numpy(gt_frame).float().unsqueeze(0).unsqueeze(0)) 
                    #     show_flow(evs, batch_flow['flow_final'], self.frame_warp.warp_frame(prev_image, batch_flow['flow_final'])-pred_image)
                    

                    
                    pred_image_numpy = pred_image.squeeze().detach().cpu().data.numpy()
                    pred_image_numpy = np.uint8(pred_image_numpy*255)
                    
                    if frame_idx == 1 or frame_idx % 5 == 0:
                        image_writer(pred_image_numpy, frame_idx+1)
                        event_img = make_event_preview(evs.cpu().data.numpy(), mode='grayscale', num_bins_to_show=-1)
                        event_writer(event_img, frame_idx)
                        flow_writer(batch_flow['flow_final'].squeeze().cpu().data.numpy(), frame_idx)
      
                    frame_idx += 1


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    
    parser = argparse.ArgumentParser()
    set_configs(parser)
    cfgs = parser.parse_args()
    
    if cfgs.path_to_test_data.split('/')[-1]=='240fps':
        cfgs.image_dim = [180,320] 
    if cfgs.dvs == 'samsung':
        cfgs.image_dim = [480,640] 
    
    reconstuctor = Reconstructor(cfgs, device)
    reconstuctor()
    
    
