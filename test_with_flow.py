import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import csv

from utils.image_process import normalize_image
from data_readers.video_readers import ImageReader
from e2v.e2v_model import *

from utils.configs import set_configs
from utils.data_io import ImageWriter, EvalWriter, FlowWriter, show_whole_img, show_flow
from utils.evaluate import mse, psnr, ssim, PerceptualLoss

from utils.flow_utils import FrameWarp
from loss import FlowReconLoss

print("Selected device:", torch.cuda.current_device())

class Reconstructor(nn.Module):
    def __init__(self, cfgs, device):
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
        

        self.path_to_sequences = []
        for folder_name in os.listdir(cfgs.path_to_test_data):
            if os.path.isdir(os.path.join(cfgs.path_to_test_data, folder_name)):
                self.path_to_sequences.append(os.path.join(cfgs.path_to_test_data, folder_name))
        self.path_to_sequences.sort()

        self.video_renderer = ImageReader(device=self.device)

        # initialize reconstruction network        
        if self.model_mode == 'cista-eiflow':
            self.model = DCEIFlowCistaNet(cfgs)
        elif self.model_mode == 'cista-eraft':
            self.model = ERAFTCistaNet(cfgs)
        else:
            assert self.model_mode in ['cista-eiflow', 'cista-eraft']
        

        # Load pretrained CISTA-Flow model
        if cfgs.load_epoch_for_test:
            self.model_name = cfgs.path_to_test_model.split('/')[-2]
            cfgs.path_to_test_model = cfgs.path_to_test_model + '{}_{}.pth.tar'.format(self.model_name, cfgs.load_epoch_for_test)
            self.model_name = self.model_name + '/{}'.format(cfgs.load_epoch_for_test)
        else:
            self.model_name = os.path.splitext(cfgs.path_to_test_model.split('/')[-1])[0]

        checkpoint = torch.load(cfgs.path_to_test_model, map_location='cuda:0')
        self.model.load_state_dict(checkpoint['state_dict'], strict=False) #True

        # Replace CISTA-LSTC network with specified pretrained model
        if cfgs.path_to_e2v:
            checkpoint = torch.load(cfgs.path_to_e2v, map_location='cuda:0')
            self.model.cista_net.load_state_dict(checkpoint['state_dict'], strict=True) #True
        print(self.model)
        print('Model name: ', self.model_name)
        
        self.model.to(device)
        self.model.eval()

        self.frame_warp = FrameWarp(mode=cfgs.warp_mode)
        self.loss_fn = FlowReconLoss(cfgs.image_dim, self.frame_warp, ds=cfgs.ds, is_bi=False).to(device)
        

    def forward(self):
        # torch.backends.cudnn.enabled = False
        with torch.no_grad():
            all_seq_test_results, all_seq_flow_results = [], []
            whole_test_mean, whole_flow_mean = [], []
            num_total_frames, num_total_flow_frames = 0, 0
            metric_keys = None
            for seq_id, path_to_sequence_folder in enumerate(self.path_to_sequences):
                dataset_name=path_to_sequence_folder.split('/')[-1].split('.')[0]

                if self.test_data_name is not None and dataset_name != self.test_data_name:
                    continue
                self.video_renderer.initialize(path_to_sequence_folder, self.num_load_frames)

                states = None
                prev_image = None
                flow_states = None

                image_writer = ImageWriter(cfgs, self.model_name, dataset_name)
                eval_writer = EvalWriter(cfgs, self.model_name, dataset_name)
                flow_writer = FlowWriter(cfgs, self.model_name, dataset_name)

                all_test_results = []
                all_flow_results = []
                
                frame_idx = 0
                prev_events = None
                # gt_prev_frame = None
                

                while not self.video_renderer.ending:
                    events, frame_pack, gt_frame, flow = self.video_renderer.update_event_frame_flow_pack(mode=self.test_data_mode) 
                    
                    if prev_image is None:
                        prev_image = torch.zeros([1, 1, self.image_dim[0], self.image_dim[1]], dtype=torch.float32, device=self.device)
                        input_data = dict([])
                        input_gt = dict([])
                        
                    for i, (evs, gt_prev_frame, gt_flow) in enumerate(zip(events, frame_pack, flow)):
                        evs = torch.unsqueeze(torch.from_numpy(evs), axis=0).to(self.device) 
                        gt_prev_frame = torch.from_numpy(gt_prev_frame).unsqueeze(0).unsqueeze(0).to(self.device)
                        gt_flow = torch.from_numpy(gt_flow).unsqueeze(0).to(self.device)
                        gt_frame_tensor = frame_pack[i+1] if i < len(frame_pack)-1 else gt_frame
                        gt_frame_tensor = torch.from_numpy(gt_frame_tensor).unsqueeze(0).unsqueeze(0).to(self.device)

                        input_data['event_voxel'] = evs
                        input_data['rec_img0'] = prev_image
                        
                        if cfgs.is_gt_flow:
                            input_gt['gt_flow'] = gt_flow
                            
                        
                        if self.model_mode in ['cista-eiflow']:
                            pred_image, batch_flow, states = self.model(input_data, states, input_gt)
                        elif self.model_mode in ['cista-eraft']:
                            if frame_idx == 0:
                                evs_old = torch.zeros_like(evs)
                            input_data['event_voxel_old'] = evs_old
                            pred_image, batch_flow, states = self.model(input_data, states, input_gt)
                            evs_old = evs.clone()

                        prev_image = pred_image.clone()
                        
                        # if cfgs.display_test:
                        #     # show_whole_img(evs, init_flow, pred_flow) #torch.from_numpy(gt_frame).float().unsqueeze(0).unsqueeze(0)) 
                        #     show_flow(evs, batch_flow['flow_final'], gt_flow, self.frame_warp.warp_frame(gt_prev_frame, batch_flow['flow_final'])-gt_frame_tensor, self.frame_warp.warp_frame(gt_prev_frame, gt_flow)-gt_frame_tensor) #self.backwarp(gt_prev_frame, gt_flow)-gt_frame_tensor, 
                        
                        
                    
                    
                    batch_target = dict(
                        gt_img0 = gt_prev_frame,
                        gt_img1 = gt_frame_tensor,
                        gt_flow = gt_flow,
                    )
         
                    rec_metrics, flow_metrics = self.loss_fn.evaluate(pred_image, batch_flow['flow_final'], batch_target)
                    
 
                    pred_image_uint8 = np.uint8(pred_image.squeeze().cpu().data.numpy()*255.)

                    
                    image_writer(pred_image_uint8, frame_idx+1)
                    flow_writer(batch_flow['flow_final'].squeeze().cpu().data.numpy(), frame_idx)
                    
                    
                    if frame_idx >= 3:
                        metrics = rec_metrics | flow_metrics
                        if metric_keys is None:
                            metric_keys = list(metrics.keys())
                        all_test_results.append(list(metrics.values()))
                    frame_idx += 1

                all_test_results = np.array(all_test_results)
                mean_test_results = all_test_results.mean(0)
                
                mean_results = [eval_writer.dataset_name] + list(np.array(mean_test_results).round(4)) + [len(all_test_results)]
                all_seq_test_results.append(mean_results)
                whole_test_mean.append(mean_test_results)
                num_total_frames += len(all_test_results)
                eval_results = ' '.join(['{}: {:.4f}, '.format(metric_keys[i], mean_test_results[i]) for i in range(len(metric_keys))])
                print('\nTest set {}: Average results for {:d} frames: {} \n'.format(
                    eval_writer.dataset_name, len(all_test_results), eval_results))
                
                name_results = ['Dataset'] + metric_keys + ['N_frames']
                eval_writer(name_results, mean_results)
                
            mean_all_test_results = np.array(whole_test_mean).mean(0)
            eval_results = ' '.join(['{}: {:.4f}, '.format(metric_keys[i], mean_all_test_results[i]) for i in range(len(metric_keys))])
                
            print('\n Average results for {:d} frames: {} \n'.format(
                num_total_frames, eval_results))
            
            name_results =['Dataset'] + metric_keys + ['N_frames']
            all_seq_test_results.append(['mean'] + list(np.array(mean_all_test_results).round(4)) + [num_total_frames] )

            if cfgs.test_data_name is None:
                output_folder = eval_writer.output_data_folder.split('/')[:-1]
                output_folder = '/'.join(cur_str for cur_str in output_folder)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                path_to_write_csv = os.path.join(output_folder, 'all.csv')
                with open(path_to_write_csv, 'a+', newline='') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(name_results)
                    writer.writerows(all_seq_test_results)
                f.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    
    parser = argparse.ArgumentParser()
    set_configs(parser)
    cfgs = parser.parse_args()
    
    reconstuctor = Reconstructor(cfgs, device)
    reconstuctor()
    
