import os

import torch.nn as nn
import numpy as np
import cv2
import argparse
import csv


from utils.image_process import normalize_image
from data_readers.video_readers import ImageReader
from e2v.e2v_model import * 

from utils.configs import set_configs
from utils.data_io import * 
from utils.evaluate import mse, psnr, ssim, PerceptualLoss

from spikingjelly.activation_based import functional
from utils.flow_utils import FrameWarp
from loss import ReconLoss, voxel_warping_flow_loss
import matplotlib.pyplot as plt 

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
        self.dataset = cfgs.dataset
        print('Height: ', self.image_dim[0], 'Width: ', self.image_dim[1])
        
        self.path_to_sequences = []
        for folder_name in os.listdir(cfgs.path_to_test_data):
            if os.path.isdir(os.path.join(cfgs.path_to_test_data, folder_name)):
                self.path_to_sequences.append(os.path.join(cfgs.path_to_test_data, folder_name))
        self.path_to_sequences.sort()

        self.video_renderer = ImageReader(cfgs, device=self.device)
        
        # initialize reconstruction network        
        if self.model_mode == 'cista-eiflow':
            self.model = DCEIFlowCistaNet(cfgs)
        elif self.model_mode == 'cista-eraft':
            self.model = ERAFTCistaNet(cfgs)
        elif self.model_mode == 'cista-idnet':
            self.model = IDCistaNet(cfgs)
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

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params}")
        # for float32
        total_memory_MB = total_params * 32 / 8 / 1024 / 1024
        print(f"Estimated model memory size: {total_memory_MB:.2f} MB")

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('\n Number of parameters in {}: {:d}'.format(self.model_mode, params))

        self.frame_warp = FrameWarp(mode=cfgs.warp_mode)
        self.loss_fn = ReconLoss(self.frame_warp).to(device)
        

        
    def forward(self):
        # torch.backends.cudnn.enabled = False
        with torch.no_grad():
            all_seq_test_results = []
            whole_test_mean = []
            num_total_frames = 0
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
                event_writer = EventWriter(cfgs, self.model_name, dataset_name)
                warped_event_writer = EventWriter(cfgs, self.model_name, dataset_name, 'warped_events')

                all_test_results = []
                frame_idx = 0
                prev_events = None
                # gt_prev_frame = None
                while not self.video_renderer.ending:
                    events, _, gt_frame = self.video_renderer.update_event_frame_pack_fix(self.limit_num_events, self.test_data_mode) #maxmin

                    if prev_image is None:
                        prev_image = torch.zeros([1, 1, self.image_dim[0], self.image_dim[1]], dtype=torch.float32, device=self.device)

                    for i, evs in enumerate(events):
                        evs = torch.unsqueeze(torch.from_numpy(evs), axis=0).to(self.device) 

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
                        elif self.model_mode in ['cista-idnet']:
                            if frame_idx == 0:
                                flow_init = None
                            pred_image, batch_flow, states = self.model(input_data, states, flow_init)
                            flow_init = batch_flow['next_flow']
                            
                        prev_image = pred_image.clone()
                        
                        # if cfgs.display_test:
                        #     # show_whole_img(evs, init_flow, pred_flow) #torch.from_numpy(gt_frame).float().unsqueeze(0).unsqueeze(0)) 
                        #     show_flow(evs, batch_flow['flow_final'], self.frame_warp.warp_frame(prev_image, batch_flow['flow_final'])-pred_image)

       
                    gt_image_norm = torch.from_numpy(gt_frame).unsqueeze(0).unsqueeze(0).to(self.device)
                    if self.dataset == 'ECD':
                        gt_image_norm = normalize_image(gt_image_norm, 0, 100) #--------
        
                    rec_metrics = self.loss_fn.evaluate(pred_image, gt_image_norm) #pred_image
                    FWL = voxel_warping_flow_loss(evs, batch_flow['flow_final'])/ voxel_warping_flow_loss(evs, torch.zeros_like(batch_flow['flow_final']))
                    
                    pred_image_uint8 = np.uint8(255*pred_image.squeeze().cpu().data.numpy()) # HQF
                    

                    if frame_idx==0 or (frame_idx+1) %1 == 0:
                        image_writer(pred_image_uint8, frame_idx+1) #-------
                        flow_writer(batch_flow['flow_final'].squeeze().cpu().data.numpy(), frame_idx)
                        event_img = make_event_preview(evs.cpu().data.numpy(), mode='red-blue', num_bins_to_show=-1) #mode='grayscale'
                        event_writer(event_img, frame_idx)
                        # event_img = make_event_preview(add_image0['voxel_grid_warped'].cpu().data.numpy(), mode='grayscale', num_bins_to_show=1) #mode='red-blue'
                        # event_img = make_event_preview(add_image['voxel_grid_warped'].cpu().data.numpy(), mode='grayscale', num_bins_to_show=1) #mode='grayscale'
                        # warped_event_writer(event_img, frame_idx)
                    
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
    
