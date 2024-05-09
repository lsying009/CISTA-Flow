import os
os.environ["KMP_BLOCKTIME"] = "0"
import sys
import h5py
import numpy as np

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
import torch.utils.data.dataset as dataset

from .MVSEC_utils import generate_corresponding_gt_flow, eventsToVoxel, eventsToTXYP
# from utils.augmentor import fetch_augmentor
# from utils.event_uitls import eventsToVoxel
from utils.event_process import events_to_voxel_grid, event_preprocess


DatasetMapping = {
    'in1': 'indoor_flying/indoor_flying1',
    'inday1': 'indoor_flying/indoor_flying1',
    'indoor1': 'indoor_flying/indoor_flying1',
    'indoor_flying1': 'indoor_flying/indoor_flying1',

    'in2': 'indoor_flying/indoor_flying2',
    'inday2': 'indoor_flying/indoor_flying2',
    'indoor2': 'indoor_flying/indoor_flying2',
    'indoor_flying2': 'indoor_flying/indoor_flying2',

    'in3': 'indoor_flying/indoor_flying3',
    'inday3': 'indoor_flying/indoor_flying3',
    'indoor3': 'indoor_flying/indoor_flying3',
    'indoor_flying3': 'indoor_flying/indoor_flying3',

    'in4': 'indoor_flying/indoor_flying4',
    'inday4': 'indoor_flying/indoor_flying4',
    'indoor4': 'indoor_flying/indoor_flying4',
    'indoor_flying4': 'indoor_flying/indoor_flying4',

    'out1': 'outdoor_day/outdoor_day1',
    'outday1': 'outdoor_day/outdoor_day1',
    'outdoor1': 'outdoor_day/outdoor_day1',
    'outdoor_day1': 'outdoor_day/outdoor_day1',

    'out2': 'outdoor_day/outdoor_day2',
    'outday2': 'outdoor_day/outdoor_day2',
    'outdoor2': 'outdoor_day/outdoor_day2',
    'outdoor_day2': 'outdoor_day/outdoor_day2',
}

# Valid_Time_Index = {
#     'indoor_flying/indoor_flying1': [314, 2199],
#     'indoor_flying/indoor_flying2': [314, 2199],
#     'indoor_flying/indoor_flying3': [314, 2199],
#     'indoor_flying/indoor_flying4': [196, 570],
#     'outdoor_day/outdoor_day1': [245, 3000],
#     'outdoor_day/outdoor_day2': [4375, 7002],
# }

Valid_Time_Index = {
    'indoor_flying1': [314, 2199],
    'indoor_flying2': [314, 2199],
    'indoor_flying3': [314, 1870], #[314, 2199]
    'indoor_flying4': [196, 570],
    'outdoor_day1': [245, 3000],
    'outdoor_day2': [4375, 7002],
}


class MVSEC(dataset.Dataset):
    def __init__(self, args, data_root, data_split='in1', data_mode='full', train_ratio=0.6, skip_num=None, aug_params=None):
        super().__init__()
        self.args = args

        self.args.crop_size = [256, 256]
        self.data_root = data_root
        self.data_split = data_split
        assert data_split in DatasetMapping.keys()
        self.data_filepath = os.path.join(
            data_root, data_split + '_data.hdf5') #DatasetMapping[data_split]
        self.gt_filepath = os.path.join(
            data_root, data_split + '_gt.hdf5') #DatasetMapping[data_split]
        print(self.data_filepath, self.gt_filepath)
        assert os.path.isfile(self.data_filepath)
        assert os.path.isfile(self.gt_filepath)

        self.data_mode = data_mode
        self.train_ratio = train_ratio

        self.event_bins = args.num_bins
        self.event_polarity = False # if args.no_event_polarity else True

        # self.augmentor = None
        # if aug_params is not None:
        #     self.augmentor = fetch_augmentor(is_event=True, is_sparse=True, aug_params=aug_params)
        # 'skip images in dataset to get more events'
        args.skip_num = 1
        if skip_num is None:
            self.skip_num = args.skip_num
        else:
            self.skip_num = skip_num

        # 'continue' or 'interrupt' or 'skip by events number'
        # 'skip images in dataset to get more events', not working
        args.skip_mode = 'c'
        if args.skip_mode == 'continue' or args.skip_mode == 'c':
            self.skip_mode = 'c'
        elif args.skip_mode == 'interrupt' or args.skip_mode == 'i':
            self.skip_mode = 'i'
        else:
            raise NotImplementedError("skip mode {} is not supported!".format(args.skip_mode))

        self.raw_index_shift = Valid_Time_Index[data_split][0]
        self.raw_index_max = Valid_Time_Index[data_split][1] - 1
        if self.skip_mode == 'i':
            self.data_length = (self.raw_index_max -
                                self.raw_index_shift) // self.skip_num - 1
        elif self.skip_mode == 'c':
            self.data_length = self.raw_index_max - \
                self.raw_index_shift - (self.skip_num - 1)

        np.random.seed(20)
        split_index = np.random.rand(self.data_length) <= self.train_ratio
        if self.data_mode == 'full':
            self.INDEX_MAP = [i for i in range(self.data_length)]
        elif self.data_mode == 'train':
            self.INDEX_MAP = [i for i in range(self.data_length) if split_index[i] ]
        elif self.data_mode == 'val':
            self.INDEX_MAP = [i for i in range(self.data_length) if not split_index[i] ]
        else:
            raise NotImplementedError("unknow data mode {}".format(self.data_mode))
        self.data_length = len(self.INDEX_MAP)

    def open_hdf5(self):

        data_file = h5py.File(self.data_filepath, 'r')
        self.events_data = data_file.get('davis/left/events')
        self.image_data = data_file.get('davis/left/image_raw')
        self.image_ts_data = data_file.get('davis/left/image_raw_ts')
        self.image_event_inds = data_file.get('davis/left/image_raw_event_inds')
        assert len(self.image_data) == len(self.image_ts_data)

        gt_file = h5py.File(self.gt_filepath, 'r')
        self.flow_dist_data = gt_file.get('davis/left/flow_dist')
        self.flow_dist_ts = gt_file.get('davis/left/flow_dist_ts')
        self.flow_dist_ts_numpy = np.array(self.flow_dist_ts, dtype=np.float32)

        self.image_length = len(self.image_data)
        self.event_length = len(self.events_data)
        self.flow_length = len(self.flow_dist_data)
        
        assert self.data_length <= self.image_length

    def __getitem__(self, index):
    
        if not hasattr(self, 'events_data'):
            self.open_hdf5()

        if self.skip_mode == 'i':
            raw_index = self.INDEX_MAP[index] * self.skip_num + self.raw_index_shift
        elif self.skip_mode == 'c':
            raw_index = self.INDEX_MAP[index] + self.raw_index_shift
        assert raw_index < self.raw_index_max

        image1 = self.image_data[raw_index]
        image1_ts = self.image_ts_data[raw_index]
        image1_event_index = self.image_event_inds[raw_index]
        image2 = self.image_data[raw_index + self.skip_num]
        image2_ts = self.image_ts_data[raw_index + self.skip_num]
        image2_event_index = self.image_event_inds[raw_index + self.skip_num]
        assert image1_event_index < image2_event_index
        assert image2_event_index < self.event_length

        if self.skip_mode == 'i' or self.skip_mode == 'c':
            events = self.events_data[image1_event_index:image2_event_index]
            next_ts = image2_ts

        height, width = image1.shape[:2]
        event_voxel = eventsToVoxel(events, num_bins=self.event_bins, height=height, width=width, \
            event_polarity=self.event_polarity, temporal_bilinear=True)

        flow_left_index = np.searchsorted(self.flow_dist_ts_numpy, image1_ts, side='right') - 1
        flow_right_index = np.searchsorted(self.flow_dist_ts_numpy, next_ts, side='right')
        assert flow_left_index <= flow_right_index
        assert flow_left_index < self.flow_length
        assert flow_right_index < self.flow_length

        flows = self.flow_dist_data[flow_left_index:flow_right_index]
        flows_ts = self.flow_dist_ts_numpy[flow_left_index:flow_right_index+1]

        final_flow = generate_corresponding_gt_flow(flows, flows_ts, image1_ts, next_ts)
        final_flow = final_flow.transpose(1, 2, 0)

        # grayscale images
        if len(image1.shape) == 2:
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        crop_height, crop_width = self.args.crop_size[:2]
        if 'out' in self.data_split:
            assert crop_height < height and crop_width < width
            start_y = (height - crop_height) // 2
            start_x = (width - crop_width) // 2
            image1 = image1[start_y:start_y+crop_height, start_x:start_x+crop_width, :]
            image2 = image2[start_y:start_y+crop_height, start_x:start_x+crop_width, :]
            event_voxel = event_voxel[:, start_y:start_y+crop_height, start_x:start_x+crop_width]
            final_flow = final_flow[start_y:start_y+crop_height, start_x:start_x+crop_width, :]

        # if self.augmentor is not None:
        #     event_voxel, image1, image2, final_flow, _, _, _ = \
        #         self.augmentor(event_voxel, image1, image2, final_flow)

        height, width = image1.shape[:2]
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        final_flow = torch.from_numpy(final_flow).permute(2, 0, 1).float()
        event_voxel = torch.from_numpy(event_voxel).float()

        event_valid = torch.norm(event_voxel, p=2, dim=0, keepdim=False) > 0
        event_valid = event_valid.float().unsqueeze(0)

        flow_valid = (torch.norm(final_flow, p=2, dim=0, keepdim=False) > 0) & (final_flow[0].abs() < 1000) & (final_flow[1].abs() < 1000)
        flow_valid = flow_valid.float().unsqueeze(0)

        if height == crop_height and width == crop_width:
            pass
        else:
            assert crop_height < height and crop_width < width
            start_y = (height - crop_height) // 2
            start_x = (width - crop_width) // 2
            image1 = image1[:, start_y:start_y+crop_height, start_x:start_x+crop_width]
            image2 = image2[:, start_y:start_y+crop_height, start_x:start_x+crop_width]
            event_voxel = event_voxel[:, start_y:start_y+crop_height, start_x:start_x+crop_width]
            event_valid = event_valid[:, start_y:start_y+crop_height, start_x:start_x+crop_width]
            final_flow = final_flow[:, start_y:start_y+crop_height, start_x:start_x+crop_width]
            flow_valid = flow_valid[:, start_y:start_y+crop_height, start_x:start_x+crop_width]

        height, width = image1.shape[:2]

        basename = "{}_{:0>5d}".format(self.data_split, index)

        batch = dict(
            gt_img0=image1,
            gt_img1=image2,
            event_voxel=event_voxel,
            gt_flow=final_flow,
            flow_valid=flow_valid,
        )

        # batch = dict(
        #     index=index,
        #     raw_index=raw_index,
        #     basename=basename,
        #     height=height,
        #     width=width,
        #     image1=image1,
        #     image2=image2,
        #     event_voxel=event_voxel,
        #     event_valid=event_valid,
        #     flow_gt=final_flow,
        #     flow_valid=flow_valid,
        # )
        return batch
    
    def get_raw_events(self, index):

        if not hasattr(self, 'events_data'):
            self.open_hdf5()

        if self.skip_mode == 'i':
            raw_index = self.INDEX_MAP[index] * self.skip_num + self.raw_index_shift
        elif self.skip_mode == 'c' or self.skip_mode == 'e':
            raw_index = self.INDEX_MAP[index] + self.raw_index_shift
        assert raw_index < self.raw_index_max

        image1_event_index = self.image_event_inds[raw_index]
        image2_event_index = self.image_event_inds[raw_index + self.skip_num]
        assert image1_event_index < image2_event_index
        assert image2_event_index < self.event_length

        if self.skip_mode == 'i' or self.skip_mode == 'c':
            events = self.events_data[image1_event_index:image2_event_index]

        return events

    def __len__(self):
        return self.data_length


class MVSEC_NE(dataset.Dataset):
    def __init__(self, args, data_root, data_split='in1', data_mode='full', train_ratio=0.6, skip_num=None, aug_params=None):
        super().__init__()
        '''Load events according to fixed number of events'''
        self.args = args

        self.width = 346
        self.height = 260
        self.args.crop_size = [260, 346] #[256, 256]
        self.data_root = data_root
        self.data_split = data_split
        assert data_split in DatasetMapping.keys()
        self.data_filepath = os.path.join(
            data_root, data_split + '_data.hdf5')
        self.gt_filepath = os.path.join(
            data_root, data_split + '_gt.hdf5')
        print(self.data_filepath, self.gt_filepath)
        assert os.path.isfile(self.data_filepath)
        assert os.path.isfile(self.gt_filepath)

        self.data_mode = data_mode
        self.train_ratio = train_ratio

        self.num_events = args.num_events
        self.event_bins = args.num_bins
        self.event_polarity = False # if args.no_event_polarity else True

        args.skip_num = 1
        if skip_num is None:
            self.skip_num = args.skip_num
        else:
            self.skip_num = skip_num


        args.skip_mode = 'i'
        if args.skip_mode == 'continue' or args.skip_mode == 'c':
            self.skip_mode = 'c'
        elif args.skip_mode == 'interrupt' or args.skip_mode == 'i':
            self.skip_mode = 'i'
        else:
            raise NotImplementedError("skip mode {} is not supported!".format(args.skip_mode))

        self.raw_index_shift = Valid_Time_Index[data_split][0]
        self.raw_index_max = Valid_Time_Index[data_split][1] - 1
        
        
        if self.skip_mode == 'i':
            self.data_length = (self.raw_index_max -
                                self.raw_index_shift) // self.skip_num - 1
        elif self.skip_mode == 'c':
            self.data_length = self.raw_index_max - \
                self.raw_index_shift - (self.skip_num - 1)

        np.random.seed(20)
        split_index = np.random.rand(self.data_length) <= self.train_ratio
        if self.data_mode == 'full':
            self.INDEX_MAP = [i for i in range(self.data_length)]
        elif self.data_mode == 'train':
            self.INDEX_MAP = [i for i in range(self.data_length) if split_index[i] ]
        elif self.data_mode == 'val':
            self.INDEX_MAP = [i for i in range(self.data_length) if not split_index[i] ]
        else:
            raise NotImplementedError("unknow data mode {}".format(self.data_mode))
        self.data_length = len(self.INDEX_MAP)

    def open_hdf5(self):

        data_file = h5py.File(self.data_filepath, 'r')
        self.events_data = data_file.get('davis/left/events')
        self.image_data = data_file.get('davis/left/image_raw')
        self.image_ts_data = data_file.get('davis/left/image_raw_ts')
        self.image_event_inds = data_file.get('davis/left/image_raw_event_inds')
        assert len(self.image_data) == len(self.image_ts_data)

        gt_file = h5py.File(self.gt_filepath, 'r')
        self.flow_dist_data = gt_file.get('davis/left/flow_dist')
        self.flow_dist_ts = gt_file.get('davis/left/flow_dist_ts')
        self.flow_dist_ts_numpy = np.array(self.flow_dist_ts, dtype=np.float32)

        self.image_length = len(self.image_data)
        self.event_length = len(self.events_data)
        self.flow_length = len(self.flow_dist_data)
        # print(self.image_length, self.event_length, self.flow_length)
        assert self.data_length <= self.image_length

    def events_to_voxel(self, events, height, width):
        evs = events_to_voxel_grid(events, 
                        num_bins=self.event_bins,
                        width=width,
                        height=height)
        # evs = torch.from_numpy(evs).float()
        crop_height, crop_width = self.args.crop_size[:2]
        start_y = (height - crop_height) // 2
        start_x = (width - crop_width) // 2
        evs = evs[:, start_y:start_y+crop_height, start_x:start_x+crop_width]
        evs = event_preprocess(evs, mode='std', filter_hot_pixel=True)
        
        evs = torch.from_numpy(evs).unsqueeze(0).float()
        return evs
        # event_voxel_list.append([evs, event_window[i].shape[0]])

    
    def __getitem__(self, index):
    
        if not hasattr(self, 'events_data'):
            self.open_hdf5()

        if self.skip_mode == 'i':
            raw_index = self.INDEX_MAP[index] * self.skip_num + self.raw_index_shift
        elif self.skip_mode == 'c':
            raw_index = self.INDEX_MAP[index] + self.raw_index_shift
        assert raw_index < self.raw_index_max

        image1 = self.image_data[raw_index]
        image1_ts = self.image_ts_data[raw_index]
        image1_event_index = self.image_event_inds[raw_index]
        image2 = self.image_data[raw_index + self.skip_num]
        image2_ts = self.image_ts_data[raw_index + self.skip_num]
        image2_event_index = self.image_event_inds[raw_index + self.skip_num]
        assert image1_event_index < image2_event_index
        assert image2_event_index < self.event_length

        if self.skip_mode == 'i' or self.skip_mode == 'c':
            # x, y, t, p
            events = self.events_data[image1_event_index:image2_event_index]
            events = eventsToTXYP(events)
            next_ts = image2_ts
        
        # self.height, self.width = image1.shape[:2]
        if self.width is None:
            self.height, self.width = image1.shape[:2]
            
        # print('num_events: ', events.shape[0])
        NE = self.num_events if self.num_events > 0 else events.shape[0]
        num_evs = round(events.shape[0] / NE)
        if num_evs == 0:
            num_evs = 1
        event_window = np.array_split(events, num_evs, axis=0)
        
        event_voxel_list = []
        raw_events_list = []
        for i in range(num_evs):
            # evs = events_to_voxel_grid(event_window[i], 
            #                 num_bins=self.event_bins,
            #                 width=self.width,
            #                 height=self.height)
            # evs = event_preprocess(evs, mode='std', filter_hot_pixel=False)
            # evs = torch.from_numpy(evs).float()
            # event_voxel_list.append([evs, event_window[i].shape[0]])
            raw_events_list.append([event_window[i], event_window[i].shape[0]])

            
            # event_voxel = eventsToVoxel(events, num_bins=self.event_bins, height=height, width=width, \
            #     event_polarity=self.event_polarity, temporal_bilinear=True)

        flow_left_index = np.searchsorted(self.flow_dist_ts_numpy, image1_ts, side='right') - 1
        flow_right_index = np.searchsorted(self.flow_dist_ts_numpy, next_ts, side='right')
        assert flow_left_index <= flow_right_index
        assert flow_left_index < self.flow_length
        assert flow_right_index < self.flow_length

        flows = self.flow_dist_data[flow_left_index:flow_right_index]
        flows_ts = self.flow_dist_ts_numpy[flow_left_index:flow_right_index+1]
        
        final_flow = generate_corresponding_gt_flow(flows, flows_ts, image1_ts, next_ts)
        final_flow = final_flow.transpose(1, 2, 0)

        # grayscale images
        if len(image1.shape) == 2:
            image1 = image1[..., None]
            image2 = image2[..., None]
            # image2 = np.tile(image2[..., None], (1, 1, 1))
        else:
            image1 = image1[..., :3].mean(-1, keepdim=True)
            image2 = image2[..., :3].mean(-1, keepdim=True)

        crop_height, crop_width = self.args.crop_size[:2]
        # if 'out' in self.data_split:
        #     assert crop_height < height and crop_width < width
        #     start_y = (height - crop_height) // 2
        #     start_x = (width - crop_width) // 2
        #     image1 = image1[start_y:start_y+crop_height, start_x:start_x+crop_width, :]
        #     image2 = image2[start_y:start_y+crop_height, start_x:start_x+crop_width, :]
        #     # event_voxel = event_voxel[:, start_y:start_y+crop_height, start_x:start_x+crop_width]
        #     final_flow = final_flow[start_y:start_y+crop_height, start_x:start_x+crop_width, :]


        # self.height, self.width = image1.shape[:2]
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()/255.
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()/255.
        
        final_flow = torch.from_numpy(final_flow).permute(2, 0, 1).float()

        # event_voxel = torch.from_numpy(event_voxel).float()

        # event_valid = torch.norm(event_voxel, p=2, dim=0, keepdim=False) > 0
        # event_valid = event_valid.float().unsqueeze(0)

        flow_valid = (torch.norm(final_flow, p=2, dim=0, keepdim=False) > 0) & (final_flow[0].abs() < 1000) & (final_flow[1].abs() < 1000)
        flow_valid = flow_valid.float().unsqueeze(0)

        if self.height == crop_height and self.width == crop_width:
            pass
        else:
            assert crop_height < self.height and crop_width < self.width
            start_y = (self.height - crop_height) // 2
            start_x = (self.width - crop_width) // 2
            image1 = image1[:, start_y:start_y+crop_height, start_x:start_x+crop_width]
            image2 = image2[:, start_y:start_y+crop_height, start_x:start_x+crop_width]
            # event_voxel = event_voxel[:, start_y:start_y+crop_height, start_x:start_x+crop_width]
            # event_voxel_list = [[evs[:, start_y:start_y+crop_height, start_x:start_x+crop_width], NE] for (evs, NE) in event_voxel_list]
            # event_valid = event_valid[:, start_y:start_y+crop_height, start_x:start_x+crop_width]
            final_flow = final_flow[:, start_y:start_y+crop_height, start_x:start_x+crop_width]
            flow_valid = flow_valid[:, start_y:start_y+crop_height, start_x:start_x+crop_width]

        basename = "{}_{:0>5d}".format(self.data_split, index)

        batch = dict(
            gt_img0=image1,
            gt_img1=image2,
            org_width = self.width,
            org_height = self.height,
            # event_voxel=event_voxel,
            gt_flow=final_flow,
            flow_valid=flow_valid,
        )

        # return event_voxel_list, batch
        return raw_events_list, batch
    
    def get_raw_events(self, index):

        if not hasattr(self, 'events_data'):
            self.open_hdf5()

        if self.skip_mode == 'i':
            raw_index = self.INDEX_MAP[index] * self.skip_num + self.raw_index_shift
        elif self.skip_mode == 'c' or self.skip_mode == 'e':
            raw_index = self.INDEX_MAP[index] + self.raw_index_shift
        assert raw_index < self.raw_index_max

        image1_event_index = self.image_event_inds[raw_index]
        image2_event_index = self.image_event_inds[raw_index + self.skip_num]
        assert image1_event_index < image2_event_index
        assert image2_event_index < self.event_length

        if self.skip_mode == 'i' or self.skip_mode == 'c':
            events = self.events_data[image1_event_index:image2_event_index]

        return events

    def __len__(self):
        return self.data_length
