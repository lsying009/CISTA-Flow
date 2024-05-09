import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data 
import cv2
from utils.event_process import *


class TrainfusedEventData(data.Dataset):
    '''Training sequence loader
        Load data sequence to train E2V reconstruction model
        The number of events per reconstruction is limited to ~<limit_num_events> (15000 as default)
        
        The format of train_data_txt: 
        seq_id num_events timestamp_0(in seconds) timestamp_1 path_to_frame_0 path_to_frame_1 path_to_events between frame_0 and frame_1

    '''
    def __init__(self, train_data_txt, cfgs):
        self.txt_file = train_data_txt
        self.path_to_train_data = cfgs.path_to_train_data
        self.num_bins = cfgs.num_bins
        self.height, self.width = cfgs.image_dim
        self.limit_num_events = cfgs.num_events
        self.len_sequence = cfgs.len_sequence
        self.is_reverse_events = cfgs.is_reverse_events 
        self.warp_mode = cfgs.warp_mode
        self.is_bi = cfgs.is_bi
        self.model_mode = cfgs.model_mode
        
        # GT flow is based on backward warping, requrie reverse for forward warping
        if cfgs.is_forward_flow:
            self.flow_name = "flow01"
            self.flow_name_bw = "flow10"
        else:
            self.flow_name = "flow10"
            self.flow_name_bw = "flow01"
        self.flow_coef = -1 if self.warp_mode == 'forward' else 1 

        
        self.video_cnt = []
        self.event_paths = []
        self.image_paths = []
        self.next_image_paths = []
        self.num_events_list = []
        self.flow_paths = []
        
        self.to_tensor = transforms.ToTensor()

        
        with open(self.txt_file,'rb') as f:
            for line in f:
                str_list = line.strip().split()
                self.video_cnt.append(int(str_list[0])) #video_cnt
                self.num_events_list.append(int(str_list[1]))
                self.image_paths.append(str(str_list[4], encoding = "utf-8")) #cur_img_path1
                self.next_image_paths.append(str(str_list[5], encoding = "utf-8")) #cur_next_img_path
                self.event_paths.append(str(str_list[6], encoding = "utf-8"))
                self.flow_paths.append(str(str_list[7], encoding = "utf-8"))
        f.close()
        
        if self.len_sequence > 0:
            self.split_sequences()
        else:
            self.split_sequences_keep_org_seq()

    
    def __len__(self):
        return len(self.sequence_line_id)   
    
    def split_sequences(self):
        prev_video_id = -1
        self.sequence_line_id = []
        line_id_per_sequence = []
        for line_id, video_id in enumerate(self.video_cnt):
            if video_id != prev_video_id and video_id!=0:
                self.sequence_line_id.append(line_id_per_sequence)
                line_id_per_sequence = []
                prev_video_id = video_id
                
            line_id_per_sequence.append(line_id)
        if line_id_per_sequence:
            self.sequence_line_id.append(line_id_per_sequence)


    def split_sequences_keep_org_seq(self):
        prev_video_id = -1
        sum_num_events = 0
        self.sequence_line_id = []
        line_id_per_reconstruction = []
        line_id_per_sequence = []
        frame_cnt, single_frame_cnt = 0, 0
        for line_id, video_id in enumerate(self.video_cnt):
            if video_id != prev_video_id and video_id !=0:
                if line_id_per_reconstruction:
                    line_id_per_sequence.append(line_id_per_reconstruction)
                self.sequence_line_id.append(line_id_per_sequence)
                line_id_per_sequence = []
                line_id_per_reconstruction = []
                prev_video_id = video_id
                sum_num_events = 0
                single_frame_cnt = 0
                frame_cnt = 0
                
            cur_num_event = self.num_events_list[line_id]
            sum_num_events += cur_num_event
            line_id_per_reconstruction.append(line_id)
            single_frame_cnt += 1
            if sum_num_events >= self.limit_num_events or (single_frame_cnt==1 and sum_num_events > 0.8*self.limit_num_events):
                line_id_per_sequence.append(line_id_per_reconstruction)
                frame_cnt += 1
                sum_num_events = 0
                single_frame_cnt = 0
                line_id_per_reconstruction = []
        # if line_id_per_reconstruction:
        #     line_id_per_sequence.append(line_id_per_reconstruction)
        if line_id_per_sequence:
            self.sequence_line_id.append(line_id_per_sequence)
    
    
    def _e2_voxelgrid(self, event_patch, mode='std', is_reverse=False):
        event_patch = events_to_voxel_grid(event_patch, 
                                            num_bins=self.num_bins,
                                            width=self.width,
                                            height=self.height,
                                            is_reverse=is_reverse)
        event_patch = event_preprocess(event_patch, mode=mode,  filter_hot_pixel=False)

        event_patch = torch.from_numpy(event_patch)
        return event_patch
    
    
    def __getitem__(self, index):
        line_id_per_sequence = self.sequence_line_id[index]
        prev_event_patch = None
        seq_events = []
        seq_flow = []
        seq_gt_img = []
        seq_img = []
        seq_batch = []
        for line_id in line_id_per_sequence:
            event_window = np.empty((0,4),dtype=np.float32)
            event_path = os.path.join(self.path_to_train_data, self.event_paths[line_id])
            event_window = np.load(event_path, allow_pickle=True) #["arr_0"]
            event_window = np.stack((event_window["t"], event_window["x"], event_window["y"],event_window["p"]), axis=1)

            # [N,4] (t,x,y,p) --> [N,4] 
            event_patch = self._e2_voxelgrid(event_window, mode='std', is_reverse=False)  #mode='maxmin'
            
            
            if self.is_reverse_events: #is_load_event_pair
                reversed_event_patch = self._e2_voxelgrid(event_window, mode='std', is_reverse=True)  #mode='maxmin'

            
            if self.model_mode == 'cista-eraft':
                if line_id ==0 or int(self.event_paths[line_id].split('.')[-2].split('_')[-1])==0:
                    event_patch_old = torch.zeros_like(event_patch)
                else:    
                    event_path = os.path.join(self.path_to_train_data, self.event_paths[line_id-1])
                    event_window_old = np.load(event_path, allow_pickle=True) #["arr_0"]
                    event_window_old = np.stack((event_window_old["t"], event_window_old["x"], event_window_old["y"],event_window_old["p"]), axis=1)
                    event_patch_old = self._e2_voxelgrid(event_window_old, is_reverse=False)
            
            flow_path = os.path.join(self.path_to_train_data, self.flow_paths[line_id])
            if not self.is_bi:
                gt_flow = self.flow_coef * torch.from_numpy(np.load(flow_path, allow_pickle=True)[self.flow_name])
                # seq_flow.append(gt_flow)
            else:
                gt_flow = self.flow_coef * torch.from_numpy(np.load(flow_path, allow_pickle=True)[self.flow_name])
                gt_flow_bw = self.flow_coef * torch.from_numpy(np.load(flow_path, allow_pickle=True)[self.flow_name_bw])


            img = np.float32((cv2.imread(os.path.join(self.path_to_train_data, self.image_paths[line_id]), cv2.IMREAD_GRAYSCALE))/255.0)   
            img = self.to_tensor(img)
            
            gt_img = np.float32((cv2.imread(os.path.join(self.path_to_train_data, self.next_image_paths[line_id]), cv2.IMREAD_GRAYSCALE))/255.0)
            gt_img = self.to_tensor(gt_img)

            
            batch_data = dict(
                    event_voxel=event_patch,
            )
            batch_target = dict(
                    gt_img0=img,
                    gt_img1=gt_img,
                    gt_flow=gt_flow,
            )
            
            
            if self.is_reverse_events:
                batch_data['event_voxel_bw'] = event_voxel_bw
            if self.is_bi:
                batch_target['gt_flow_bw'] = gt_flow_bw
            if self.model_mode == 'cista-eraft':
                batch_data['event_voxel_old'] = event_patch_old

            seq_batch.append((batch_data, batch_target))
        return seq_batch

