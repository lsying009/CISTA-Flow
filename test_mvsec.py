'''
    The GT flow in MVSEC datasets is not suitable for comparing with our estimation
'''

import os
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import cv2
import argparse
import csv


from utils.image_process import normalize_image
from data_readers.video_readers import ImageReader
from e2v.e2v_model import * 

from utils.configs import set_configs
from utils.data_io import ImageWriter, EvalWriter, FlowWriter, show_whole_img, show_flow

from data_readers.MVSEC import MVSEC_NE

from utils.flow_utils import FrameWarp
from loss import ReconLoss, FlowReconLoss


test_data_list = ['indoor_flying1', 'indoor_flying2', 'outdoor_day1', 'outdoor_day2'] # 'indoor_flying3',


class Reconstructor(nn.Module):
    def __init__(self, cfgs, device):
        super(Reconstructor, self).__init__()
        self.cfgs = cfgs
        self.image_dim = cfgs.image_dim
        self.reader_type = cfgs.reader_type
        self.model_mode = cfgs.model_mode
        self.device = device
        self.num_load_frames = cfgs.test_img_num
        self.test_data_name = cfgs.test_data_name
        self.warp_mode = cfgs.warp_mode

        self.limit_num_events = cfgs.num_events #int(self.image_dim[0] * self.image_dim[1] *0.35)
        self.test_data_name = cfgs.test_data_name #'indoor_flying1'

        
        self.path_to_sequences = []
        self.path_to_seq_names = []
        if self.test_data_name is None:
            for folder_name in os.listdir(cfgs.path_to_test_data):
                data_folder = os.path.join(cfgs.path_to_test_data, folder_name)
                if os.path.isdir(data_folder):
                    for data_file in os.listdir(data_folder):
                        data_name = data_file.split('.')[0].split('_data')[0]
                        if os.path.isfile(os.path.join(data_folder, data_file)) and \
                            data_name in test_data_list and data_name not in self.path_to_seq_names:
                            self.path_to_sequences.append(data_folder) 
                            self.path_to_seq_names.append(data_name)
        else:
            self.path_to_seq_names.append(self.test_data_name)
            for folder_name in os.listdir(cfgs.path_to_test_data):
                data_folder = os.path.join(cfgs.path_to_test_data, folder_name)
                if os.path.isdir(data_folder):
                    if data_folder.split('/')[-1] in self.test_data_name:
                        self.path_to_sequences.append(data_folder)
               
        self.path_to_sequences.sort()
        self.path_to_seq_names.sort()
        
        print(self.path_to_sequences, self.path_to_seq_names)
        
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

        print(self.model)
        print('Model name: ', self.model_name)
        
        self.model.to(device)
        self.model.eval()

        self.frame_warp = FrameWarp(mode=cfgs.warp_mode)
        self.loss_fn = ReconLoss(self.frame_warp, device=device)
        # self.loss_fn = FlowReconLoss(cfgs.image_dim, self.frame_warp, is_bi=False).to(device)
        
    def forward(self):
        # torch.backends.cudnn.enabled = False
        with torch.no_grad():
            all_seq_test_results = []
            whole_test_mean = []
            num_total_frames = 0
            metric_keys = None
            for seq_id, (path_to_sequence_folder, data_name) in enumerate(zip(self.path_to_sequences, self.path_to_seq_names)):

                test_data = MVSEC_NE(self.cfgs, data_root=path_to_sequence_folder, data_split=data_name)
                self.video_renderer = data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)       
                
                states = None
                prev_image = None
                flow_states = None

                image_writer = ImageWriter(cfgs, self.model_name, data_name) #+'_gt'
                eval_writer = EvalWriter(cfgs, self.model_name, data_name)
                flow_writer = FlowWriter(cfgs, self.model_name, data_name)

                all_test_results = []
                
                frame_idx = 0
                prev_events = None

                
                print('data_length', len(self.video_renderer))

                num_events_per_recon = 0
                events_per_rec = []
                for batch_idx, (raw_events_list, batch_gt) in enumerate(self.video_renderer):
                    if batch_idx >= self.num_load_frames:
                        break
                    org_width, org_height = batch_gt['org_width'].squeeze().data.numpy(), batch_gt['org_height'].squeeze().data.numpy()
                    batch_gt = {key: value.to(self.device) for key, value in batch_gt.items()}
                    gt_frame_tensor = batch_gt['gt_img1']
                    if prev_image is None:
                        prev_image = torch.zeros([1, 1, self.image_dim[0], self.image_dim[1]], dtype=torch.float32, device=self.device)
                    
                    
                    for i, (events, N_E) in enumerate(raw_events_list):
                        num_events_per_recon += N_E.squeeze().data.numpy()
                        events_per_rec.append(events.squeeze().data.numpy())
                        if self.limit_num_events>0 and num_events_per_recon < 0.8*self.limit_num_events:
                            continue
                        else:
                            # print('NE: ', num_events_per_recon)
                            num_events_per_recon = 0
                            evs = self.video_renderer.dataset.events_to_voxel(np.concatenate(events_per_rec, axis=0), org_height, org_width)
                            events_per_rec = []
                            
                            evs = evs.to(self.device) 
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

                            if cfgs.display_test:
                                show_flow(evs, batch_flow['flow_final'], self.frame_warp.warp_frame(prev_image, batch_flow['flow_final'])-pred_image)
                            prev_image = pred_image.clone()

                    if num_events_per_recon !=0:
                        continue
                

                    rec_metrics = self.loss_fn.evaluate(pred_image, gt_image_norm) #pred_image
                    FWL = voxel_warping_flow_loss(evs, batch_flow['flow_final'])/ voxel_warping_flow_loss(evs, torch.zeros_like(batch_flow['flow_final']))
                    
                    # rec_metrics, flow_metrics = self.loss_fn.evaluate(pred_image, batch_flow['flow_final'], batch_gt)
                
                    pred_image_numpy = pred_image.squeeze().detach().cpu().data.numpy()
                    pred_image_numpy = np.uint8(cv2.normalize(pred_image_numpy, None, 0, 255, cv2.NORM_MINMAX)) # HQF
   
                    if frame_idx==0 or (frame_idx+1)%10 == 0:
                        image_writer(pred_image_numpy, frame_idx+1)
                        flow_writer(batch_flow['flow_final'].squeeze().cpu().data.numpy(), frame_idx)

                    if frame_idx >=3:
                        if metric_keys is None:
                            metric_keys = list(rec_metrics.keys())+['FWL']
                        all_test_results.append(list(rec_metrics.values())+[FWL.cpu().data.numpy()])
                        
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

            # name_results = ['Dataset', 'MSE', 'PSNR', 'SSIM', 'LPIPS', 'N_frames']
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
    
