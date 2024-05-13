import torch.nn as nn
from .base_layers import *
from utils.data_io import show_whole_img
from DCEIFlow.DCEIFlow import DCEIFlow
from ERAFT.eraft import ERAFT
from idn.idedeq import IDEDEQIDO
from omegaconf import OmegaConf
from utils.flow_utils import FrameWarp

class CistaLSTCNet(nn.Module):
     def __init__(self, image_dim, base_channels=64, depth=5, num_bins=5):
          super(CistaLSTCNet, self).__init__()
          '''
               CISTA-LSTC network for events-to-video reconstruction
          '''
          self.num_bins = num_bins
          self.depth = depth
          self.height, self.width = image_dim
          self.num_states = 3 
          
          self.We = ConvLayer(in_channels=self.num_bins, out_channels=int(base_channels/2), kernel_size=3,\
          stride=1, padding=1, groups=1) #We_new
          
           
          self.Wi = ConvLayer(in_channels=1, out_channels=int(base_channels/2), kernel_size=3,\
               stride=1, padding=1) 
          self.W0 = ConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3,\
               stride=2, padding=1)
          
          

          self.P0 = ConvLSTC(x_size=base_channels, z_size=2*base_channels, output_size=2*base_channels, kernel_size=3) 

          lista_block = IstaBlock(base_channels=base_channels, is_recurrent=False) 
          self.lista_blocks = nn.ModuleList([lista_block for i in range(self.depth)])
          

          self.Dg = RecurrentConvLayer(in_channels=2*base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1,
               activation='relu')

          self.upsamp_conv = UpsampleConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=0, activation='relu') #activation='relu'
          
          self.final_conv = ConvLayer(in_channels=base_channels, out_channels=1, \
               kernel_size=3, stride=1, padding=1)
          
          self.sigmoid = nn.Sigmoid()


     def forward(self, events, prev_image, prev_states):
          '''
          Inputs:
               events: torch.tensor, float32, [batch_size, num_bins, H, W]
                    Event voxel grid
               prev_image: torch.tensor, float32, [batch_size, 1, H, W]
                    Reconstructed frame from the last reconstruction
               prev_states: None or list of torch.tensor, float32
                    Previous states
          Outputs:
               rec_I: torch.tensor, float32, [batch_size, 1, H, W]
                    Reconstructed frame
               states: list of torch.tensor, float32
                    Updated states in e2v_net
          '''
          
          if prev_states is None:
               prev_states = [None]*self.num_states
          states = [] 

          x_E = self.We(events)
          x_I = self.Wi(prev_image)
          x1 = connect_cat(x_E, x_I) 

          x1 = self.W0(x1) 

          z, state = self.P0(x1, prev_states[-2], prev_states[0] if prev_states[0] is not None else None)

          states.append(state)
               
          tmp = z.clone()

          for i in range(self.depth):
               tmp = self.lista_blocks[i].D(tmp)
               x = x1- tmp
               x = self.lista_blocks[i].P(x)
               x = x + z
               z = softshrink(x, self.lista_blocks[i].Lambda) 
               tmp = z      
          states.append(z)

          rec_I, state = self.Dg(z, prev_states[-1])
          
          states.append(state)
          
          rec_I = self.upsamp_conv(rec_I)
 
          rec_I = self.sigmoid(self.final_conv(rec_I))

          return rec_I, states



class BaseFlowRec(nn.Module):
     def __init__(self, args):
          super(BaseFlowRec, self).__init__()
          # image_dim, base_channels=64, depth=5, num_bins=5, warp_mode='forward'
          self.image_dim = args.image_dim
          self.num_bins = args.num_bins
          self.frame_warp = FrameWarp(mode=args.warp_mode)
          self.fix_net_name = None

          self.scale_factor = 0.5
          self.cista_net = CistaLSTCNet(image_dim=args.image_dim, base_channels=args.base_channels, depth=args.depth, num_bins=args.num_bins)
     
          self.event_flownet = None #TODO
          
     def fix_params(self, net_name):
          '''Fix parameters of reconstruction or flow network by specifiying net_name (rec or flow) '''
          self.fix_net_name = net_name
          if net_name == 'rec':
               for param in self.cista_net.parameters():
                    param.requires_grad = False
               for param in self.event_flownet.parameters():
                    param.requires_grad = True 
               self.event_flownet.train()
          elif net_name == 'flow':
               for param in self.event_flownet.parameters():
                    param.requires_grad = False
               for param in self.cista_net.parameters():
                    param.requires_grad = True
               self.event_flownet.eval() #freeze_bn()
               self.cista_net.train()
          else:
               assert net_name in ['flow', 'rec']
   


# single GPU
class DCEIFlowCistaNet(BaseFlowRec):
     '''CISTA-Flow: CISTA-LSTC + DCEIFlow'''
     def __init__(self, args):
          super(DCEIFlowCistaNet, self).__init__(args)
          self.event_flownet = DCEIFlow(num_bins=self.num_bins, args=args) 

     def forward(self, batch_data, states, batch_gt=dict([])):
          '''
          Input:
               batch_data: dict
                    event_voxel: E_0^1
                    event_voxel_bw (optional): E_1^0 , event voxel grid with reversed time
                                             only used for bilateral training if GT I1 is known
                                             (If used, must along with gt_img1), 
                    flow_init(optional): default None, used if warm_start 
                    rec_img0: previous reconstructed frame \hat{I}_0
               states: list
                    For CISTA-LSTC, length = 3, states[1] is sparse codes Z from previous reconstruction
               batch_gt(optional): dict
                    Ground truth images or flow, only for training
                    gt_img0 (for DCEIFlow (GT I)), gt_img1 (for DCEIFlow (GT I)), gt_flow (for CISTA (GT Flow)) F_0->1
          Output:
               I_rec: Tensor, reconstructed frame \hat{I}_1
               batch_flow: dict
                    output of flow network, batch_flow['flow_final'] is the estimated forward flow \hat{F}_0->1
                    for details refer to DCEIFlow
               states: list
                    updated states
          '''
          
          # Flow estimation using E_0^1 and \hat{I}_0
          # gt_img is only used for training DCEIFlow (GT I)
          batch_flow = self.event_flownet(event_voxel=batch_data['event_voxel'], image1=batch_gt['gt_img0'] if 'gt_img0' in batch_gt.keys() else batch_data['rec_img0'], \
                                        image2= batch_gt['gt_img1'] if 'gt_img1' in batch_gt.keys() else None, \
                                        reversed_event_voxel=batch_data['event_voxel_bw'] if 'event_voxel_bw' in batch_data.keys() else None, \
                                        flow_init=batch_data['flow_init'] if 'flow_init' in batch_data.keys() else None)
          flow_final = batch_flow['flow_final']
          
          if self.fix_net_name == 'flow':
               flow_final = flow_final.detach()
               flow_final.requires_grad = False
          
          # gt_flow is only used for training CISTA (GTFlow)
          if 'gt_flow' in batch_gt.keys():
               flow_final = batch_gt['gt_flow']
          
          if not flow_final.any():
               warped_I = batch_data['rec_img0']
          else:
               # Warp inputs I and Z for CISTA-LSTC using flow
               warped_I = self.frame_warp.warp_frame(batch_data['rec_img0'], flow_final)
               if states is not None:
                    downsampled_flow = nn.functional.interpolate(flow_final, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
                    states[1] = self.frame_warp.warp_frame(states[1], downsampled_flow)
          
          # Reconstruction using E_0^1, warped I and Z
          I_rec, states = self.cista_net(batch_data['event_voxel'], warped_I, states)

          return I_rec, batch_flow, states


# single GPU
class ERAFTCistaNet(BaseFlowRec):
     '''CISTA-Flow: CISTA-LSTC + E-RAFT'''
     def __init__(self, args):
          super(ERAFTCistaNet, self).__init__(args)
          self.event_flownet = ERAFT(args)

     def forward(self, batch_data, states, batch_gt=dict([])): #, gt_prev_frame=None, gt_frame=None, gt_flow=None):
          '''
          Input:
               batch_data: dict
                    A pair of event voxel grids, event_voxel_old and event_voxel
               states: list
                    For CISTA-LSTC, length = 3, states[1] is sparse codes Z from previous reconstruction
               batch_gt(Optional): dict
                    gt_flow (F_0->1): Ground truth flow, only for training CISTA (GT Flow)
          Output:
               I_rec: Tensor, reconstructed frame \hat{I}_1
               batch_flow: dict
                    output of flow network, batch_flow['flow_final'] is the estimated forward flow \hat{F}_0->1
                    for details refer to ERAFT
               states: list
                    updated states
          '''
          
          # Flow estimation using E_0^1 and \hat{I}_0
          batch_flow = self.event_flownet(image1=batch_data['event_voxel_old'], image2=batch_data['event_voxel'])
          flow_final = batch_flow['flow_final']
          
          if self.fix_net_name == 'flow':
               flow_final = flow_final.detach()
               flow_final.requires_grad = False
          
          # gt_flow is only used for training CISTA (GTFlow)
          if 'gt_flow' in batch_gt.keys():
               flow_final = batch_gt['gt_flow']
               
          if not flow_final.any():
               warped_I = batch_data['rec_img0']
          else:
               # Warp inputs I and Z for CISTA-LSTC using flow
               warped_I = self.frame_warp.warp_frame(batch_data['rec_img0'], flow_final)
               if states is not None:
                    downsampled_flow = nn.functional.interpolate(flow_final, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
                    states[1] = self.frame_warp.warp_frame(states[1], downsampled_flow)
          
          # Reconstruction using E_0^1, warped I and Z
          I_rec, states = self.cista_net(batch_data['event_voxel'], warped_I, states)

          return I_rec, batch_flow, states


# single GPU
class IDCistaNet(BaseFlowRec):
     '''CISTA-Flow: CISTA-LSTC + IDNet'''
     def __init__(self, args):
          super(IDCistaNet, self).__init__(args)
          config = {
               'update_iters': 1,
               'pred_next_flow': True,
               'image_dim':args.image_dim,
          }
          config = OmegaConf.create(config)
          self.event_flownet = IDEDEQIDO(config)
        

     def forward(self, batch_data, states, flow_init=None, batch_gt=dict([])): #, gt_prev_frame=None, gt_frame=None, gt_flow=None):
          '''
          Input:
               batch_data: dict
                    A pair of event voxel grids, event_voxel_old and event_voxel
               states: list
                    For CISTA-LSTC, length = 3, states[1] is sparse codes Z from previous reconstruction
               batch_gt(Optional): dict
                    gt_flow (F_0->1): Ground truth flow, only for training CISTA (GT Flow)
          Output:
               I_rec: Tensor, reconstructed frame \hat{I}_1
               batch_flow: dict
                    output of flow network, batch_flow['flow_final'] is the estimated forward flow \hat{F}_0->1
                    for details refer to ERAFT
               states: list
                    updated states
          '''
          
          # Flow estimation using E_0^1 and \hat{I}_0
          batch_flow = self.event_flownet(event_bins=batch_data['event_voxel'], flow_init=flow_init)
          # flow_init = batch_flow['next_flow']
          flow_final = batch_flow['flow_final']
          
          if self.fix_net_name == 'flow':
               flow_final = flow_final.detach()
               flow_final.requires_grad = False
          
          # gt_flow is only used for training CISTA (GTFlow)
          if 'gt_flow' in batch_gt.keys():
               flow_final = batch_gt['gt_flow']
               
          if not flow_final.any():
               warped_I = batch_data['rec_img0']
          else:
               # Warp inputs I and Z for CISTA-LSTC using flow
               warped_I = self.frame_warp.warp_frame(batch_data['rec_img0'], flow_final)
               if states is not None:
                    downsampled_flow = nn.functional.interpolate(flow_final, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
                    states[1] = self.frame_warp.warp_frame(states[1], downsampled_flow)
          
          # Reconstruction using E_0^1, warped I and Z
          I_rec, states = self.cista_net(batch_data['event_voxel'], warped_I, states)

          return I_rec, batch_flow, states
       
     
# 2 GPU
class DCEIFlowCistaNet2GPU(BaseFlowRec):
     '''CISTA-Flow: CISTA-LSTC + DCEIFlow
          Two-GPU version for training if out of memory using 1 GPU
          CISTA-LSTC and DCEIFlow on different GPUs
     '''
     
     def __init__(self, args):
          super(DCEIFlowCistaNet2GPU, self).__init__(args)
          self.cista_net.to('cuda:0')
          self.event_flownet = DCEIFlow(num_bins=self.num_bins, args=args).to('cuda:1')
  
     def forward(self, batch_data, states, batch_gt): #loss_mode, is_consis,, gt_prev_frame=None, gt_frame=None, gt_flow=None):
          batch_flow = self.event_flownet(event_voxel=batch_data['event_voxel'].to('cuda:1'), image1=batch_data['rec_img0'].to('cuda:1'), \
                                        image2= batch_gt['gt_img1'].clone().to('cuda:1') if 'gt_img1' in batch_gt.keys() else None, \
                                        reversed_event_voxel=batch_data['event_voxel_bw'].to('cuda:1') if 'event_voxel_bw' in batch_data.keys() else None, \
                                        flow_init=batch_data['flow_init'].to('cuda:1') if 'flow_init' in batch_data.keys() else None)

          batch_flow = dict_to_device(batch_flow, 'cuda:0')
          flow_final = batch_flow['flow_final'].detach()
          
          if self.fix_net_name == 'flow':
               flow_final.requires_grad = False
  
          if 'gt_flow' in batch_gt.keys():
               flow_final = batch_gt['gt_flow']

          warped_I = self.frame_warp.warp_frame(batch_data['rec_img0'], flow_final) #.to('cuda:0')
          if states is not None:
               downsampled_flow = nn.functional.interpolate(flow_final, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
               states[1] = self.frame_warp.warp_frame(states[1], downsampled_flow)

          I_rec, states = self.cista_net(batch_data['event_voxel'], warped_I, states) 


          return I_rec, batch_flow, states
 

def dict_to_device(dictionary, device): 
     assert isinstance(dictionary, dict) 
     new_dict = {}  
     for key, value in dictionary.items():  
          if isinstance(value, dict):  
               new_value = dict_to_device(value, device)  
          elif isinstance(value, (list, tuple)):  
               new_value = [tensor.to(device) for tensor in value] 
          else: # isinstance(value, torch.Tensor):  
               new_value = value.to(device)  
          new_dict[key] = new_value  
     return new_dict


def list_to_device(list_values, device): 
     # assert isinstance(dictionary, (list, tuple)) 
     new_list = []  
     for value in list_values:   
          if isinstance(value, (list, tuple)):  
               new_value = [tensor.to(device) for tensor in value] 
          else: # isinstance(value, torch.Tensor):  
               new_value = value.to(device)  
          new_list.append(new_value) 
     return new_list


