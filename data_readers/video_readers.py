import torch
import numpy as np
import cv2
import os

from superslomo.upsamp_sequence import Upsampler
from utils.event_process import event_preprocess, events_to_voxel_grid, events_to_voxel_grid_pol
from .event_readers import SingleEventReaderNpz, RefTimeEventReaderZip, FixedSizeEventReader


def read_timestamps_file(path_to_timestamps, unit='s'):
    ''' Read timestamps from a .txt file
        # TODO customise your according to the format of timestamps
        unit: str, 's' / 'us' / 'ms'
            Time unit of the time in the file, should rescaled to seconds [s]
        Output: List of float
            List timestamps in seconds

    '''
    timestamps = []
    if path_to_timestamps.split('/')[-1] == 'timestamps.txt':
        with open(path_to_timestamps, 'r') as f:
            for line in f:
                timestamps.append(float(line.strip().split()[1]))
        f.close()
    else:
        with open(path_to_timestamps, 'r') as f:
            for line in f:
                timestamps.append(float(line.strip().split()[0]))
                
        f.close()
    # t0 = timestamps[0]
    timestamps = np.array(timestamps)
    if unit in ['us']:
        timestamps /= 1e6
    elif unit in ['ns']:
        timestamps /= 1e9
                
    return list(timestamps)


## parent class for video reader
class VR:
    def __init__(self, image_dim, num_bins=5,  device="cuda:0"):
        ''' Video_reader
        Load video sequences in a frame pack for each reconstruction
        image_dim: [height, width]
        
        '''
        self.height, self.width = image_dim
        self.prev_ts_cache = np.zeros(1, dtype=np.float64)
        self.frame_id = 0
        self.num_frames = -1
        self.timestamps = []
        self.device = device
        ## Params for events
        self.num_bins = num_bins
        self.ending = False

    def update_frame(self):
        ## TODO redefine
        return np.zeros((self.height, self.width),dtype=np.uint8), 0
    
    def update_flow(self):
        ## TODO redefine
        return np.zeros((2, self.height, self.width),dtype=np.uint8), 0
        
    def update_events(self):
        return
    
    def update_event_frame_pack(self, limit_num_events=-1, mode='upsampled'):
        '''
        Load GT frame and corresponding events for one reconstruction
        Between frames but set upper bound of event count per reconstruction (using limit_num_events)
        Inputs:
            limit_num_events: int, default: -1
                limited number of events per reconstruction
            mode: str, 'real' or 'upsampled', default: upsampled
                'real': real data sequences, load events between two consecutive frames and split them by limit_num_events
                'upsampled': simulated data sequences with high frame rate, load events between several frames to reach limit_num_events
        Outputs: 
            event_windows: list of event voxel, np.array B x H x W
            frame_pack: list of frames,  np.array, H x W
            gt_frame: np.ndarray [height, width]
                Ground truth image for the reconstruction (the last frame in frame_pack)
        '''
        frame_pack = []
        ## Skip the first frame
        if self.frame_id == 0:
            self.prev_frame, _  = self.update_frame()
        frame_pack.append(self.prev_frame)
        if limit_num_events > 0 and mode == 'upsampled': 
            ## for simulated data 
            sum_num_events = 0
            event_pack = []
            while (sum_num_events < 0.8*limit_num_events) and (self.frame_id<self.num_frames):
                gt_frame, _  = self.update_frame()
                events = self.update_events()
                frame_pack.append(gt_frame)
                if events is not None:
                    event_pack.append(events)
                    sum_num_events += len(events)
                if len(event_pack)>1:
                    event_window = np.concatenate(event_pack, 0)
                else:
                    event_window = event_pack[0]
            frame_pack.pop(-1)
            self.prev_frame = gt_frame
        else:    
            gt_frame, _  = self.update_frame()
            event_window = self.update_events()
            self.prev_frame = gt_frame
            # frame_pack.append(gt_frame)
            if event_window is None:
                event_window = []
        # print(self.frame_id, len(event_window), self.num_frames)
        if self.frame_id >= self.num_frames:
            self.ending = True
        # print(event_window)
        self.num_events = len(event_window)
        event_windows = []
        if limit_num_events <= 0 or mode == 'upsampled':
            event_window = events_to_voxel_grid(event_window, 
                                                num_bins=self.num_bins,
                                                width=self.width,
                                                height=self.height)
            event_window = event_preprocess(event_window, filter_hot_pixel=False)
            event_windows.append(event_window)
        else: # mode == 'real', or with limit_num_events, the number of events ususally exceeds limit_num_events
            num_evs = round(event_window.shape[0] / limit_num_events)
            if num_evs == 0:
                num_evs = 1
            event_window = np.array_split(event_window, num_evs, axis=0)
            
            for i in range(num_evs):
                evs = events_to_voxel_grid(event_window[i], 
                                num_bins=self.num_bins,
                                width=self.width,
                                height=self.height)
                evs = event_preprocess(evs, filter_hot_pixel=True)
                event_windows.append(evs)
                
        return event_windows, frame_pack, gt_frame


    def update_event_frame_pack_fix(self, limit_num_events=-1, mode='upsampled'):
        '''
        Load GT frame and corresponding events for one reconstruction
        at fixed number of events
        Inputs:
            limit_num_events: int, default: -1
                target number of events per reconstruction
            mode: str, 'real' or 'upsampled', default: upsampled
                Here must be 'upsampled'
                'real': real data sequences, load events between two consecutive frames and split them by limit_num_events
                'upsampled': simulated data sequences with high frame rate, load events between several frames to reach limit_num_events
        Outputs: 
            event_windows: list of event voxel, np.array B x H x W
            frame_pack: list of frames,  np.array, H x W
            gt_frame: np.ndarray [height, width]
                Ground truth image for the reconstruction (the last frame in frame_pack)
        '''
        frame_pack = []
        ## Skip the first frame
        if self.frame_id == 0:
            self.prev_frame, _  = self.update_frame()
        frame_pack.append(self.prev_frame)
        if limit_num_events > 0 and mode == 'upsampled': 
            ## for simulated data 
            sum_num_events = 0
            event_pack = []
            while (sum_num_events < 0.8*limit_num_events) and (self.frame_id<self.num_frames):
                gt_frame, _  = self.update_frame()
                events = self.update_events()
                frame_pack.append(gt_frame)
                if events is not None:
                    event_pack.append(events)
                    sum_num_events += len(events)
                if len(event_pack)>1:
                    event_window = np.concatenate(event_pack, 0)
                else:
                    event_window = event_pack[0]
            frame_pack.pop(-1)
            self.prev_frame = gt_frame
        elif limit_num_events > 0 and mode == 'real': 
            sum_num_events = 0
            event_pack = []
            while (sum_num_events < limit_num_events) and (self.frame_id<self.num_frames):   
                gt_frame, _  = self.update_frame()
                events = self.update_events()
                
                if events is not None:
                    event_pack.append(events)
                    sum_num_events += len(events)
                if len(event_pack)>1:
                    event_window = np.concatenate(event_pack, 0)
                else:
                    event_window = event_pack[0]
                if self.frame_id >= self.num_frames:
                    self.ending = True
            self.prev_frame = gt_frame
        else: #limit_num_events < 0
            gt_frame, _  = self.update_frame()
            event_window = self.update_events()
            if self.frame_id >= self.num_frames:
                self.ending = True
            self.prev_frame = gt_frame
            
        event_window = event_window[event_window[:,1]<self.width]
        event_window = event_window[event_window[:,2]<self.height]
        
        self.num_events = len(event_window)
        event_windows = []
        if limit_num_events <= 0 or mode == 'upsampled':
            event_window = events_to_voxel_grid(event_window, 
                                                num_bins=self.num_bins,
                                                width=self.width,
                                                height=self.height)
            event_window = event_preprocess(event_window, filter_hot_pixel=False)
            event_windows.append(event_window)
        else: # mode == 'real', or with limit_num_events, the number of events ususally exceeds limit_num_events
            num_evs = round(event_window.shape[0] / limit_num_events)
            if num_evs == 0:
                num_evs = 1
            event_window = np.array_split(event_window, num_evs, axis=0)
            
            for i in range(num_evs):
                evs = events_to_voxel_grid(event_window[i], 
                                num_bins=self.num_bins,
                                width=self.width,
                                height=self.height)
                evs = event_preprocess(evs, filter_hot_pixel=True)
                event_windows.append(evs)
                
        return event_windows, frame_pack, gt_frame


    def update_event_frame_flow_pack(self, mode='upsampled'):
        '''
        Load GT frame and corresponding events for one reconstruction
        Inputs:
            mode: str, 'real' or 'upsampled', default: upsampled
                Here must be 'upsampled'
                'real': real data sequences, load events between two consecutive frames and split them by limit_num_events
                'upsampled': simulated data sequences with high frame rate, load events between several frames to reach limit_num_events
        Outputs: 
            event_windows: list of event voxel, np.array B x H x W
            frame_pack: list of frames,  np.array, H x W
            gt_frame: np.ndarray [height, width]
                Ground truth image for the reconstruction (the last frame in frame_pack)
            flow_list: list of flow, np.array, 2 x H x W
        '''
        assert mode == 'upsampled', "Data mode can not be 'real'!"
        
        frame_pack = []
        ## Skip the first frame
        if self.frame_id == 0:
            self.prev_frame, _  = self.update_frame()
        frame_pack.append(self.prev_frame)
   
        gt_frame, _  = self.update_frame()
        flow = self.update_flow(self.prev_frame, gt_frame)
        self.prev_frame = gt_frame
        event_window = self.update_events()
        if event_window is None:
            event_window = []
        
        if self.frame_id >= self.num_frames:
            self.ending = True

        self.num_events = len(event_window)
        event_windows = []
        flow_list = []
        # if limit_num_events <= 0 or mode == 'upsampled':
        event_window = events_to_voxel_grid(event_window, 
                                            num_bins=self.num_bins,
                                            width=self.width,
                                            height=self.height)
        event_window = event_preprocess(event_window, filter_hot_pixel=False)
        event_windows.append(event_window)
        flow_list.append(flow)
      
        return event_windows,frame_pack, gt_frame, flow_list

    


class VideoReader(VR):
    def __init__(self, image_dim, ds=[1/4,1/4]):
        super(VideoReader, self).__init__(image_dim)
        '''Read HFR video in video format'''
        self.ds = ds
        
    def initialize(self, path_to_video, num_load_frames=-1):
        ''' Initialize / reset variables for a video sequence 
            Read all the frames and timestamps from the video
            num_load_frames: total number of frames to be loaded in the sequence
        '''
        cap = cv2.VideoCapture(path_to_video)
    
        if (cap.isOpened()== False):
            assert "Error opening video stream or file"
                
        self.frames, self.timestamps = [], []
        frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        num_load_frames = frame_number if num_load_frames < 0 else num_load_frames
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        while(cap.isOpened()):
            frame_exists, frame = cap.read()
            if frame_exists:
                if frame_count > num_load_frames:
                    break
                # timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                self.timestamps.append(float(frame_count)/fps)
                
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, dsize=(int(gray.shape[1]*self.ds[1]), int(gray.shape[0]*self.ds[0])))
                if frame.shape[0] > frame.shape[1]:
                    gray = gray.T
                self.frames.append(gray)
                
            else:
                break
        
        cap.release()
        self.num_frames = len(self.frames)

        self.prev_ts_cache.fill(0)
        self.frame_id = 0
        
    def update_frame(self, frame_id=None): 
        if frame_id is not None:
            self.frame_id = frame_id
        frame = self.frames[self.frame_id]
        timestamp = self.timestamps[self.frame_id]
        self.frame_id += 1
    
        return frame, timestamp
    
             

class ImageReader(VR):
    def __init__(self, cfgs, device='cuda:0'):
        super(ImageReader, self).__init__(cfgs.image_dim, cfgs.num_bins, device)
        '''Load HFR video in image format from a folder'''
        self.time_unit = cfgs.time_unit
        self.is_with_flow = cfgs.is_with_flow
        self.is_forward_flow = cfgs.is_forward_flow
        self.flow_name = "flow01" if cfgs.is_forward_flow else "flow10"
        self.flow_coef = -1 if not cfgs.is_forward_flow else 1
        self.warp_mode = cfgs.warp_mode
        self.dataset = cfgs.dataset
        
        
    def initialize(self, path_to_sequence, num_load_frames=-1):
        ''' Initialise / reset variables for a video sequence 
            Read all the frames and timestamps from the folder (path_to_sequence)
            num_load_frames: total number of frames to be loaded in the sequence
        '''
        self.frame_id = 0
        self.event_id = 0
        self.flow_id = 0
        self.ending = False
        self.prev_frame = None
        
        self.path_to_frames = []
        path_to_events = []
        self.path_to_flow = []
        
        flow_folder_name = 'flow'
        
        for root, dirs, files in os.walk(path_to_sequence):
            # print(root, root.split('/')[-1])
            data_type_folder_name = root.split('/')[-1]
            for file_name in files:
                if file_name.split('.')[-1] in ['jpg','png']:
                    self.path_to_frames.append(os.path.join(root, file_name))
                elif file_name in ['timestamps.txt', 'images.txt', 'timestamp.txt']:
                    path_to_timestamps = os.path.join(root, file_name)
                elif ((file_name.split('.')[-1] in ['npz']) and 'flow' not in file_name) or file_name in ['events.txt', 'events.zip', 'events.csv']: 
                    path_to_events.append(os.path.join(root, file_name))
                elif self.is_with_flow and (file_name.split('.')[-1] in ['npz']) and flow_folder_name in root.split('/')[-1]: #------------
                    self.path_to_flow.append(os.path.join(root, file_name))
        self.path_to_frames.sort()
        self.path_to_flow.sort()
        # print(path_to_events)
        self.timestamps = []
        self.timestamps = read_timestamps_file(path_to_timestamps, self.time_unit)
        
            
        if num_load_frames > 0:
            self.path_to_frames = self.path_to_frames[:num_load_frames]
            self.timestamps = self.timestamps[:num_load_frames]
            if self.is_with_flow:
                self.path_to_flow = self.path_to_flow[:num_load_frames]

        if self.dataset == 'HSERGB':
            self.path_to_frames = [self.path_to_frames[0]] + self.path_to_frames
            self.timestamps = [self.timestamps[0]] + self.timestamps
        
        demo_image = cv2.imread(self.path_to_frames[0], cv2.IMREAD_GRAYSCALE)
        # print(self.path_to_frames[0], demo_image.shape)
        height = (demo_image.shape[0] // 2) * 2
        width = (demo_image.shape[1]//2) *2 
        
        
        assert height == self.height or width == self.width, "Image dim should be H{}xW{}".format(height, width)

        assert self.is_with_flow and len(self.path_to_flow) == 0, "GT Flow is not available!"
        
        self.num_frames = len(self.path_to_frames)
        
        self.prev_ts_cache = np.zeros(1, dtype=np.float64)
        
        ## if load events, define event loader
        if len(path_to_events)>1:
            path_to_events.sort()
            if num_load_frames > 0:
                path_to_events = path_to_events[:num_load_frames]
            self.event_window_iterator = SingleEventReaderNpz(path_to_events)
        elif len(path_to_events)==1:
            path_to_events = path_to_events[0]
            self.event_window_iterator = RefTimeEventReaderZip(path_to_events, self.timestamps)

    def update_frame(self, frame_id=None): 
        if frame_id is not None:
            self.frame_id = frame_id
        frame = cv2.imread(self.path_to_frames[self.frame_id], cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
        frame = frame[:self.height, :self.width]
        timestamp = self.timestamps[self.frame_id]
        
        self.frame_id += 1
        
        return frame, timestamp

    def update_flow(self, I0=None, I1=None, flow_id=None): 
        if flow_id is not None:
            self.flow_id = flow_id
        
        # flow 1 0
        flow = np.load(self.path_to_flow[self.flow_id], allow_pickle=True)[self.flow_name] #["flow10"]
        flow = self.flow_coef * flow[:self.height, :self.width]
        
        
        self.flow_id += 1
        
        return flow

    def update_events(self):
        try:
            event_window = next(self.event_window_iterator)
        except StopIteration:
            event_window = None
        self.event_id += 1
        # print('event_id: ', self.event_id)
        return event_window

            

