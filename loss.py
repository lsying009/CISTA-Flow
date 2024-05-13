# Adapted from 
# https://github.com/danqu130/DCEIFlow 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.evaluate import PerceptualLoss
from pytorch_msssim import SSIM
from utils.image_process import ImagePadder



class PSNR(nn.Module):
    def __init__(self, data_range=1): #
        super().__init__()
        self.data_range = data_range
        
    def forward(self, imgs1, imgs2):
        mse = ((imgs1/1.0 - imgs2/1.0) ** 2 ).mean()
        if mse < 1.0e-10:
            return 100
        return 20 * torch.log10(self.data_range / torch.sqrt(mse))


def voxel_warping_flow_loss(voxel, displacement, output_images=False, reverse_time=False):
    """ Adapted from:
        Temporal loss, as described in Eq. (2) of the paper 'Learning Blind Video Temporal Consistency',
        Lai et al., ECCV'18.

        This function takes an optic flow tensor and uses this to warp the channels of an
        event voxel grid to an image. The variance of this image is the resulting loss.
        :param voxel: [N x C x H x W] input voxel
        :param displacement: [N x 2 x H x W] displacement map from previous flow tensor to current flow tensor
    """
    if reverse_time:
        displacement = -displacement
    # if displacement.all():
    #     voxel_grid_warped_save = voxel.detach()
    #     voxel_grid_warped = voxel.sum(1)
    # else:
    v_shape = voxel.size()
    t_width, t_height, t_channels = v_shape[3], v_shape[2], v_shape[1]
    yy, xx = torch.meshgrid(torch.arange(t_height), torch.arange(t_width))  # xx, yy -> WxH
    xx, yy = xx.to(voxel.device).float(), yy.to(voxel.device).float()

    displacement_x = displacement[:, 0, :, :]  # N x H x W
    displacement_y = displacement[:, 1, :, :]  # N x H x W
    displacement_increment = 1.0/(t_channels-1.0)
    voxel_grid_warped = torch.zeros((v_shape[0], 1, t_height, t_width), dtype=voxel.dtype, device=voxel.device) 
    # print(voxel_grid_warped.shape, xx.shape, displacement_x.shape, displacement_y.shape)
    voxel_grid_warped_save = torch.zeros((v_shape[0], t_channels, t_height, t_width), dtype=voxel.dtype, device=voxel.device) 
    for i in range(t_channels):
        warp_magnitude_ratio = (1.0-i*displacement_increment) if reverse_time else i*displacement_increment
        #Add displacement to the x coords
        warping_grid_x = xx + displacement_x*warp_magnitude_ratio # N x H x W
        #Add displacement to the y coords
        warping_grid_y = yy + displacement_y*warp_magnitude_ratio # N x H x W
        warping_grid = torch.stack([warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2
        #Normalize the warping grid to between -1 and 1 (necessary for grid_sample API)
        warping_grid[:,:,:,1] = (2.0*warping_grid[:,:,:,1])/(t_height)-1.0 #(2.0*warping_grid[:,:,:,1])/(t_height-1)-1.0
        warping_grid[:,:,:,0] = (2.0*warping_grid[:,:,:,0])/(t_width)-1.0
        # print(xx.max(), xx.min(), yy.max(),yy.min())
        # print(warping_grid[:,:,:,1].max(), warping_grid[:,:,:,1].min(), warping_grid[:,:,:,0].max(), warping_grid[:,:,:,0].min())
        voxel_channel_warped = F.grid_sample(voxel, warping_grid,  align_corners=True) #, padding_mode='reflection'
        # print(voxel_channel_warped[:,i:i+1], voxel[:,i:i+1])
        # print(voxel_channel_warped.shape, voxel.shape, ((displacement_x*warp_magnitude_ratio)!=0).sum(), ((displacement_y*warp_magnitude_ratio)!=0).sum(), (voxel_channel_warped[:,i:i+1]-voxel[:,i:i+1]).abs().mean())
        voxel_grid_warped+=voxel_channel_warped[:, i:i+1, :, :].detach()
        voxel_grid_warped_save[:, i:i+1, :,:] = voxel_channel_warped[:, i:i+1, :, :].detach()

    variance = voxel_grid_warped.var()
    tc_loss = variance
    # tc_loss = -variance
    # if not reverse_time:
    #     reverse_tc_loss = voxel_warping_flow_loss(voxel, displacement, output_images=False, reverse_time=True)
    #     tc_loss += reverse_tc_loss
    if output_images:
        additional_output = {'voxel_grid': voxel,
                             'voxel_grid_warped': voxel_grid_warped_save} #voxel_grid_warped
        return tc_loss, additional_output
    else:
        return tc_loss


def epe(flow_pred, flow_gt, valid_gt=None):

    epe = torch.sum((flow_pred - flow_gt)**2, dim=1).sqrt()
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    epe = epe.view(-1)
    mag = mag.view(-1)

    outlier = (epe > 3.0).float()
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()

    if valid_gt is not None:
        val = valid_gt.view(-1) >= 0.5
        metrics = {
            'epe': epe[val].mean(),
            '1px': (epe[val] < 1).float().mean(),
            '3px': (epe[val] < 3).float().mean(),
            '5px': (epe[val] < 5).float().mean(),
            'F1': out[val].mean() * 100,
            'ol': outlier[val].mean() * 100,
        }
    else:
        metrics = {
            'epe': epe.mean(),
            '1px': (epe < 1).float().mean(),
            '3px': (epe < 3).float().mean(),
            '5px': (epe < 5).float().mean(),
            'F1': out.mean() * 100,
            'ol': outlier.mean() * 100,
        }
    return metrics


class FlowL1LossDict(nn.Module):
    def __init__(self, image_dim, frame_warper, ds=8, is_bi=False): #
        super().__init__()
        self.gamma = 0.8
        self.isbi = is_bi
        self.max_flow = 400
        self.warp_fn = frame_warper
        self.image_padder = ImagePadder(image_dim, min_size=32)


    def resizeflow_tosize(self, flow, new_size, mode='bilinear'):
        if new_size[0] == flow.shape[2] and new_size[1] == flow.shape[3]:
            return flow
        return F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


    def compute(self, flow_preds, flow_gt, gt_img0, gt_img1, valid_original, fmap2_gt=None, fmap2_pseudo=None):

        flow_loss = 0.0
        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        # valid = (valid_original >= 0.5) & (mag < self.max_flow)
        valid = valid_original * (mag < self.max_flow).float()

        for i in range(len(flow_preds)):
            i_weight = self.gamma**(len(flow_preds) - i - 1)
            if flow_gt.shape == flow_preds[i].shape:
                i_loss = (flow_preds[i] - flow_gt).abs()
                photo_loss = (self.warp_fn.warp_frame(gt_img0,flow_preds[i]) - gt_img1).abs()
                flow_loss += i_weight * (valid * i_loss).mean()
                flow_loss += i_weight * photo_loss.mean()
            else:
                scaled_flow_gt = self.resizeflow_tosize(flow_gt, flow_preds[i].shape[2:])
                scaled_img0 = self.resizeflow_tosize(gt_img0, flow_preds[i].shape[2:])
                scaled_img1 = self.resizeflow_tosize(gt_img1, flow_preds[i].shape[2:])
                i_loss = (flow_preds[i] - scaled_flow_gt).abs()
                photo_loss = (self.warp_fn.warp_frame(scaled_img0, flow_preds[i]) - scaled_img1).abs()
                scaled_mag = torch.sum(scaled_flow_gt**2, dim=1, keepdim=True).sqrt()
                scaled_valid = self.resizeflow_tosize(valid_original, flow_preds[i].shape[2:]) * (scaled_mag < self.max_flow).float()
                flow_loss += i_weight * (scaled_valid * i_loss).mean()
                flow_loss += i_weight * photo_loss.mean()
                

        epe = torch.sum(valid*(flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[(valid>0).view(-1)]
    
        if fmap2_pseudo is not None:
            if isinstance(fmap2_pseudo, list):
                for i in range(len(fmap2_pseudo)):
                    i_weight = self.gamma**(len(fmap2_pseudo) - i - 1) if len(fmap2_pseudo) != 1 else 1.0
                    i_loss = F.l1_loss(fmap2_pseudo[i], fmap2_gt[i]) * 10
                    pseudo_loss += i_weight * i_loss
            else:
                pseudo_loss = F.l1_loss(fmap2_pseudo, fmap2_gt) * 10

            flow_loss += pseudo_loss

        if fmap2_pseudo is None:
            metrics = {
                'flow_l1loss': flow_loss,
                'epe': epe.mean(),
                '1px': (epe < 1).float().mean(),
                '3px': (epe < 3).float().mean(),
                '5px': (epe < 5).float().mean(),
            }
        else:
            metrics = {
                'flow_l1loss': flow_loss,
                'epe': epe.mean(),
                'pseudo': pseudo_loss,
                '1px': (epe < 1).float().mean(),
                '3px': (epe < 3).float().mean(),
                '5px': (epe < 5).float().mean(),
            }

        return flow_loss, metrics

    def compute_noI2(self, flow_preds, flow_gt, gt_img0, gt_img1, valid_original):

        flow_loss = 0.0
        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        # valid = (valid_original >= 0.5) & (mag < self.max_flow)
        valid = valid_original * (mag < self.max_flow).float()

        for i in range(len(flow_preds)):
            i_weight = self.gamma**(len(flow_preds) - i - 1)
            if flow_gt.shape == flow_preds[i].shape:
                i_loss = (flow_preds[i] - flow_gt).abs()
                photo_loss = (self.warp_fn.warp_frame(gt_img0,flow_preds[i]) - gt_img1).abs()
                flow_loss += i_weight * (valid * i_loss).mean()
                flow_loss += i_weight * photo_loss.mean()
            else:
                scaled_flow_gt = self.resizeflow_tosize(flow_gt, flow_preds[i].shape[2:])
                scaled_img0 = self.resizeflow_tosize(gt_img0, flow_preds[i].shape[2:])
                scaled_img1 = self.resizeflow_tosize(gt_img1, flow_preds[i].shape[2:])
                i_loss = (flow_preds[i] - scaled_flow_gt).abs()
                # print(scaled_img0.shape, scaled_img1.shape, flow_preds[i].shape)
                photo_loss = (self.warp_fn.warp_frame(scaled_img0, flow_preds[i]) - scaled_img1).abs()
                scaled_mag = torch.sum(scaled_flow_gt**2, dim=1, keepdim=True).sqrt()
                # scaled_valid = (self.resizeflow_tosize(valid_original, flow_preds[i].shape[2:]) >= 0.5) & (scaled_mag < self.max_flow)
                scaled_valid = self.resizeflow_tosize(valid_original, flow_preds[i].shape[2:]) * (scaled_mag < self.max_flow).float()
                flow_loss += i_weight * (scaled_valid * i_loss).mean()
                flow_loss += i_weight *  photo_loss.mean()

        epe = torch.sum(valid*(flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[(valid>0).view(-1)]

        metrics = {
                'l1loss': flow_loss,
                'epe': epe.mean(),
                '1px': (epe < 1).float().mean(),
                '3px': (epe < 3).float().mean(),
                '5px': (epe < 5).float().mean(),
            }

        return flow_loss, metrics

    def evaluate(self, flow_final, batch_target):
        if 'flow_valid' in batch_target.keys():
            valid_original = batch_target['flow_valid']
        else:
            valid_original = torch.exp(-50*F.mse_loss(self.warp_fn.warp_frame(batch_target['gt_img0'], batch_target['gt_flow']), batch_target['gt_img1'], reduction='none'))
        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(batch_target['gt_flow']**2, dim=1, keepdim=True).sqrt()
        valid = valid_original * (mag < self.max_flow).float()

        # photo_loss = ((self.warp_fn.warp_frame(batch_target['gt_img0'], flow_final) - batch_target['gt_img1'])**2).mean()
        photo_loss = (self.warp_fn.warp_frame(batch_target['gt_img0'], flow_final) - batch_target['gt_img1']).abs().mean()
        epe = torch.sum(valid*(flow_final - batch_target['gt_flow'])**2, dim=1).sqrt()
        
        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        out = out.view(-1)[(valid>0).view(-1)]
        
        epe = epe.view(-1)[(valid>0).view(-1)]

        # FWL = voxel_warping_flow_loss(event_voxel_grid, flow_final)/ voxel_warping_flow_loss(event_voxel_grid, torch.zeros_like(flow_final))
        metrics = {
            'photo_loss': photo_loss.item(),
            'epe': epe.mean().item(),
            '1px': (epe > 1).float().mean().item(), # outlier
            '3px': (epe > 3).float().mean().item(),
            '5px': (epe > 5).float().mean().item(),
            'out': out.mean().item() * 100,
            # 'FWL': FWL,
        }
        return metrics

    def forward(self, out, batch_target):
        """ Loss function defined over sequence of flow predictions """
        flow_loss = 0.0
        
        flow_preds = out['flow_preds']
       
        gt_img0 = self.image_padder.pad(batch_target['gt_img0'])
        gt_img1 = self.image_padder.pad(batch_target['gt_img1'])
        flow_gt = self.image_padder.pad(batch_target['gt_flow'])
        valid = self.image_padder.pad(batch_target['valid'])

        if 'fmap2_gt' in out.keys():
            fmap2_gt = out['fmap2_gt']
            fmap2_pseudo = out['fmap2_pseudo']
        else:
            fmap2_gt = None
            fmap2_pseudo = None
        flow_loss_fw, metrics_fw = self.compute(flow_preds,flow_gt, gt_img0, gt_img1, valid, fmap2_gt, fmap2_pseudo)
        if not self.isbi:
            return flow_loss_fw, metrics_fw
        else:
            assert 'flow_preds_bw' in out.keys()
            flow_preds = out['flow_preds_bw']
            fmap2_gt = out['fmap1_gt']
            fmap2_pseudo = out['fmap1_pseudo']
            flow_gt = self.image_padder.pad(batch_target['gt_flow_bw'])
            valid = self.image_padder.pad(batch_target['valid_bw'])
            
            flow_loss_bw, metrics_bw = self.compute(flow_preds, fmap2_gt, fmap2_pseudo, flow_gt, gt_img1, gt_img0, valid)

            flow_loss = (flow_loss_fw + flow_loss_bw) * 0.5
            metrics = {}
            for key in metrics_fw:
                assert key in metrics_bw.keys()
                metrics[key] = (metrics_fw[key] + metrics_bw[key]) * 0.5

            return flow_loss, metrics


class ReconLoss(nn.Module):
    def __init__(self, frame_warper, lpips_net='alex'): #
        super().__init__()
        self.warp_fn = frame_warper
        self.lpips_loss_fn = PerceptualLoss(net=lpips_net) # alex for evaluate
        self.L1_loss_fn = nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()
        self.psnr_fn = PSNR(data_range=1)
        self.ssim_loss_fn = SSIM(data_range=1, size_average=True, channel=1, nonnegative_ssim=False) #.to(device)

    def evaluate(self, rec_img, target_img):
        lpips_loss = self.lpips_loss_fn(rec_img, target_img, normalize=True)
        mse_loss = self.mse_loss_fn(rec_img, target_img)
        ssim_loss = self.ssim_loss_fn(rec_img, target_img) 
        psnr_loss = self.psnr_fn(rec_img, target_img)
        
        metrics = {
            'mse': mse_loss.item(),
            'psnr': psnr_loss.item(),
            'ssim': ssim_loss.item(),
            'lpips': lpips_loss.item(),
        }
        return metrics


    def forward(self, out, rec_img0, batch_target, is_loss_consis=True):
        
        # M = torch.exp(-50*F.mse_loss(self.warp_fn.warp_frame(batch_target['gt_img0'], batch_target['gt_flow']), batch_target['gt_img1'], reduction='none'))
        if is_loss_consis:
            M = batch_target['valid']
            warped_prev_image = self.warp_fn.warp_frame(rec_img0, batch_target['gt_flow'])
            loss_consistency = 5*(M*F.l1_loss(warped_prev_image, out, reduction='none')).mean()
        else:
            loss_consistency = 0
        lpips_loss = self.lpips_loss_fn(out, batch_target['gt_img1'],normalize=True)
        L1_loss = self.L1_loss_fn(out, batch_target['gt_img1'])
        ssim_loss = 1 - self.ssim_loss_fn(out, batch_target['gt_img1'])
        loss = lpips_loss + L1_loss + ssim_loss + loss_consistency #+ loss_flow
        
        loss_dict = dict(
            LPIPS=lpips_loss,
            L1=L1_loss,
            SSIM=ssim_loss,
            loss_consistency=loss_consistency,
            loss_rec=loss-loss_consistency,
            loss_rec_all=loss,
        )
        
        return loss, loss_dict


class FlowReconLoss(nn.Module):
    '''
        Compute loss for reconstructed frames and estimated flows 
        Integrating  ReconLoss and FlowL1LossDict
        evaluate() for evaluation
        others for training loss
    '''
    def __init__(self, image_dim, frame_warper, ds=8, is_bi=False, lpips_net='alex'):
        super().__init__()
        self.warp_fn = frame_warper
        self.is_bi = is_bi
        self.reconstruction_loss_fn = ReconLoss(frame_warper, lpips_net=lpips_net)
        self.flow_loss_fn = FlowL1LossDict(image_dim, frame_warper, ds=ds, is_bi=is_bi)

    def evaluate(self, rec_img, flow_final, batch_target):
        rec_metrics = self.reconstruction_loss_fn.evaluate(rec_img, batch_target['gt_img1'])
        flow_metrics = self.flow_loss_fn.evaluate(flow_final, batch_target) #, event_voxel_grid)
        return rec_metrics, flow_metrics

    def compute_mask(self, batch_target):
        batch_target['valid'] = torch.exp(-50*F.mse_loss(self.warp_fn.warp_frame(batch_target['gt_img0'], batch_target['gt_flow']), batch_target['gt_img1'], reduction='none'))
        return batch_target
    
    def compute_flow_loss(self, batch_flow, batch_target, loss_mode):
        if loss_mode in ['flow', 'both']:
            if self.is_bi:
                batch_target['valid_bw'] = torch.exp(-50*F.mse_loss(self.warp_fn.warp_frame(batch_target['gt_img1'], batch_target['gt_flow_bw']), batch_target['gt_img0'], reduction='none'))
            loss, loss_dict = self.flow_loss_fn(batch_flow, batch_target)[0]
            return loss, loss_dict
        else:
            return 0., 0 #torch.tensor(0.)
    
    def compute_rec_loss(self, out, rec_img0, batch_target, loss_mode, is_loss_consis):
        if loss_mode in ['rec', 'both']:
            loss, loss_dict = self.reconstruction_loss_fn(out, rec_img0, batch_target, is_loss_consis=is_loss_consis)
            return loss, loss_dict
        else:
            return 0, 0 
    
    def forward(self, out, rec_img0, batch_flow, batch_target, loss_mode, is_loss_consis=True):
        assert loss_mode in ['rec', 'flow', 'both']
        batch_target['valid'] = torch.exp(-50*F.mse_loss(self.warp_fn.warp_frame(batch_target['gt_img0'], batch_target['gt_flow']), batch_target['gt_img1'], reduction='none'))
        loss = 0
        # loss_dict = {}
        if loss_mode in ['rec', 'both']:
            loss_rec = self.reconstruction_loss_fn(out, rec_img0, batch_target, is_loss_consis=is_loss_consis)[0]
            loss += loss_rec
            # loss_dict = {**loss_dict, **loss_rec_dict}
        if loss_mode in ['flow', 'both']:
            if self.is_bi:
                batch_target['valid_bw'] = torch.exp(-50*F.mse_loss(self.warp_fn.warp_frame(batch_target['gt_img1'], batch_target['gt_flow_bw']), batch_target['gt_img0'], reduction='none'))
            loss_flow = self.flow_loss_fn(batch_flow, batch_target)[0]
            loss += loss_flow
            # loss_dict = {**loss_dict, **loss_flow_dict}
        return loss
