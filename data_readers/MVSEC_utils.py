#!/usr/bin/env python
import numpy as np
import math
import cv2
import torch

"""
Calculates per pixel flow error between flow_pred and flow_gt. 
event_img is used to mask out any pixels without events (are 0).
If is_car is True, only the top 190 rows of the images will be evaluated to remove the hood of 
the car which does not appear in the GT.
"""
def flow_error_dense(flow_gt, flow_pred, event_img=None, is_car=False):
    max_row = flow_gt.shape[1]

    if event_img is None:
        event_img = np.ones(flow_pred.shape[0:2])
    if is_car:
        max_row = 190

    event_img_cropped = event_img[:max_row, :]
    flow_gt_cropped = flow_gt[:max_row, :, :]

    flow_pred_cropped = flow_pred[:max_row, :, :]

    event_mask = event_img_cropped > 0

    # Only compute error over points that are valid in the GT (not inf or 0).
    flow_mask = np.logical_and(
        np.logical_and(~np.isinf(flow_gt_cropped[:, :, 0]), ~np.isinf(flow_gt_cropped[:, :, 1])),
        np.linalg.norm(flow_gt_cropped, axis=2) > 0)
    total_mask = np.squeeze(np.logical_and(event_mask, flow_mask))

    gt_masked = flow_gt_cropped[total_mask, :]
    pred_masked = flow_pred_cropped[total_mask, :]

    # Average endpoint error.
    EE = np.linalg.norm(gt_masked - pred_masked, axis=-1)
    n_points = EE.shape[0]
    AEE = np.mean(EE)

    # Percentage of points with EE < 3 pixels.
    thresh = 3.
    percent_AEE = float((EE < thresh).sum()) / float(EE.shape[0] + 1e-5)

    return AEE, percent_AEE, n_points

"""
Propagates x_indices and y_indices by their flow, as defined in x_flow, y_flow.
x_mask and y_mask are zeroed out at each pixel where the indices leave the image.
The optional scale_factor will scale the final displacement.
"""
def prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
    flow_x_interp = cv2.remap(x_flow,
                              x_indices,
                              y_indices,
                              cv2.INTER_NEAREST)
    
    flow_y_interp = cv2.remap(y_flow,
                              x_indices,
                              y_indices,
                              cv2.INTER_NEAREST)

    x_mask[flow_x_interp == 0] = False
    y_mask[flow_y_interp == 0] = False
        
    x_indices += flow_x_interp * scale_factor
    y_indices += flow_y_interp * scale_factor

    return

"""
The ground truth flow maps are not time synchronized with the grayscale images. Therefore, we
need to propagate the ground truth flow over the time between two images.
This function assumes that the ground truth flow is in terms of pixel displacement, not velocity.

Pseudo code for this process is as follows:

x_orig = range(cols)
y_orig = range(rows)
x_prop = x_orig
y_prop = y_orig
Find all GT flows that fit in [image_timestamp, image_timestamp+image_dt].
for all of these flows:
  x_prop = x_prop + gt_flow_x(x_prop, y_prop)
  y_prop = y_prop + gt_flow_y(x_prop, y_prop)

The final flow, then, is x_prop - x-orig, y_prop - y_orig.
Note that this is flow in terms of pixel displacement, with units of pixels, not pixel velocity.

Inputs:
  x_flow_in, y_flow_in - list of numpy arrays, each array corresponds to per pixel flow at
    each timestamp.
  gt_timestamps - timestamp for each flow array.
  start_time, end_time - gt flow will be estimated between start_time and end time.
"""
def generate_corresponding_gt_flow(flows,
                                   flows_ts,
                                   start_time,
                                   end_time):

    flow_length = len(flows)
    # assert flow_length == len(flows_ts) - 1
    assert flow_length == len(flows_ts) - 1, f"Assertion failed: {flow_length} is not equal to {len(flows_ts) - 1}"

    x_flow = flows[0][0]
    y_flow = flows[0][1]
    gt_dt = flows_ts[1] - flows_ts[0]
    pre_dt = end_time - start_time

    # if gt_dt > pre_dt:
    if start_time > flows_ts[0] and end_time <= flows_ts[1]:
        x_flow *= pre_dt / gt_dt
        y_flow *= pre_dt / gt_dt
        return np.concatenate((x_flow[np.newaxis, :], y_flow[np.newaxis, :]), axis=0)

    x_indices, y_indices = np.meshgrid(np.arange(x_flow.shape[1]),
                                       np.arange(x_flow.shape[0]))

    x_indices = x_indices.astype(np.float32)
    y_indices = y_indices.astype(np.float32)

    orig_x_indices = np.copy(x_indices)
    orig_y_indices = np.copy(y_indices)

    # Mask keeps track of the points that leave the image, and zeros out the flow afterwards.
    x_mask = np.ones(x_indices.shape, dtype=bool)
    y_mask = np.ones(y_indices.shape, dtype=bool)

    scale_factor = (flows_ts[1] - start_time) / gt_dt
    total_dt = flows_ts[1] - start_time
    
    prop_flow(x_flow, y_flow,
              x_indices, y_indices,
              x_mask, y_mask,
              scale_factor=scale_factor)

    for i in range(1, flow_length-1):
        x_flow = flows[i][0]
        y_flow = flows[i][1]

        prop_flow(x_flow, y_flow,
                  x_indices, y_indices,
                  x_mask, y_mask)
    
        total_dt += flows_ts[i+1] - flows_ts[i]

    gt_dt = flows_ts[flow_length] - flows_ts[flow_length-1]
    pred_dt = end_time - flows_ts[flow_length-1]
    total_dt += pred_dt

    x_flow = flows[flow_length-1][0]
    y_flow = flows[flow_length-1][1]

    scale_factor = pred_dt / gt_dt

    prop_flow(x_flow, y_flow,
              x_indices, y_indices,
              x_mask, y_mask,
              scale_factor)

    x_shift = x_indices - orig_x_indices
    y_shift = y_indices - orig_y_indices
    x_shift[~x_mask] = 0
    y_shift[~y_mask] = 0

    return np.concatenate((x_shift[np.newaxis, :], y_shift[np.newaxis, :]), axis=0)



# from https://github.com/TimoStoff/event_utils/blob/master/lib/representations/voxel_grid.py

def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    """
    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)
    return img


def binary_search_torch_tensor(t, l, r, x, side='left'):
    """
    Binary search sorted pytorch tensor
    """
    if r is None:
        r = len(t)-1
    while l <= r:
        mid = l + (r - l)//2;
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r


def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
        :param xs: tensor of x coords of events
        :param ys: tensor of y coords of events
        :param ps: tensor of event polarities/weights
        :param device: the device on which the image is. If none, set to events device
        :param sensor_size: the size of the image sensor/output image
        :param clip_out_of_range: if the events go beyond the desired image size,
            clip the events to fit into the image
        :param interpolation: which interpolation to use. Options=None,'bilinear'
        :param padding if bilinear interpolation, allow padding the image by 1 to allow events to fit:
    """
    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = torch.zeros(img_size, dtype=torch.float32).to(device)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs-pxs).float()
        dys = (ys-pys).float()
        pxs = (pxs*mask).long()
        pys = (pys*mask).long()
        masked_ps = ps.squeeze()*mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        img.index_put_((ys, xs), ps.float(), accumulate=True)
    return img


def events_to_voxel_torch(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    B : number of bins in output voxel grids (int)
    device : device to put voxel grid. If left empty, same device as events
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    if isinstance(xs, np.ndarray):
        xs = torch.from_numpy(xs)
        ys = torch.from_numpy(ys)
        ts = torch.from_numpy(ts)
        ps = torch.from_numpy(ps)

    if device is None:
        device = xs.device
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))

    bins = []
    dt = ts[-1]-ts[0]
    t_norm = (ts-ts[0])/dt*(B-1)
    zeros = torch.zeros(t_norm.size())
    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = torch.max(zeros, 1.0-torch.abs(t_norm-bi))
            weights = ps*bilinear_weights
            vb = events_to_image_torch(xs, ys,
                    weights, device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        else:
            tstart = ts[0] + dt*bi
            tend = tstart + dt
            beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart) 
            end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend) 
            vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                    ps[beg:end], device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        bins.append(vb)
    bins = torch.stack(bins)
    return bins


def events_to_neg_pos_voxel_torch(xs, ys, ts, ps, B, device=None,
        sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation.
    Positive and negative events are put into separate voxel grids
    Parameters
    ----------
    xs : list of event x coordinates 
    ys : list of event y coordinates 
    ts : list of event timestamps 
    ps : list of event polarities 
    B : number of bins in output voxel grids (int)
    device : the device that the events are on
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel_pos: voxel of the positive events
    voxel_neg: voxel of the negative events
    """

    if isinstance(xs, np.ndarray):
        xs = torch.from_numpy(xs)
        ys = torch.from_numpy(ys)
        ts = torch.from_numpy(ts)
        ps = torch.from_numpy(ps)

    zero_v = torch.tensor([0.])
    ones_v = torch.tensor([1.])
    pos_weights = torch.where(ps>0, ones_v, zero_v)
    neg_weights = torch.where(ps<=0, ones_v, zero_v)

    voxel_pos = events_to_voxel_torch(xs, ys, ts, pos_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)
    voxel_neg = events_to_voxel_torch(xs, ys, ts, neg_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)

    return voxel_pos, voxel_neg


def eventsToXYTP(events, process=False):
    event_x = events[:, 0].astype(np.int32)
    event_y = events[:, 1].astype(np.int32)
    event_pols = events[:, 3].astype(np.int32)

    event_timestamps = events[:, 2]

    if process:
        last_stamp = event_timestamps[-1]
        first_stamp = event_timestamps[0]
        deltaT = last_stamp - first_stamp
        event_timestamps = (event_timestamps - first_stamp) / deltaT

    # event_pols[event_pols == 0] = -1  # polarity should be +1 / -1

    return event_x, event_y, event_timestamps, event_pols


def eventsToTXYP(events, process=False):
    event_x = events[:, 0].astype(np.int32)
    event_y = events[:, 1].astype(np.int32)
    event_pols = events[:, 3].astype(np.int32)

    event_timestamps = events[:, 2]

    if process:
        last_stamp = event_timestamps[-1]
        first_stamp = event_timestamps[0]
        deltaT = last_stamp - first_stamp
        event_timestamps = (event_timestamps - first_stamp) / deltaT

    # event_pols[event_pols == 0] = -1  # polarity should be +1 / -1

    return np.stack([event_timestamps, event_x, event_y, event_pols], axis=1)


def eventsToVoxel(events, num_bins=5, height=None, width=None, event_polarity=False, temporal_bilinear=True):
    return eventsToVoxelTorch(events, num_bins, height, width, event_polarity, temporal_bilinear).numpy()


def eventsToVoxelTorch(events, num_bins=5, height=None, width=None, event_polarity=False, temporal_bilinear=True):
    xs, ys, ts, ps = eventsToXYTP(events, process=True)

    if height is None or width is None:
        width = xs.max() + 1
        height = ys.max() + 1

    if not event_polarity:
        # generate voxel grid which has size num_bins x H x W
        voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, num_bins, sensor_size=(height, width), temporal_bilinear=temporal_bilinear)
    else:
        # generate voxel grid which has size 2*num_bins x H x W
        voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, num_bins, sensor_size=(height, width), temporal_bilinear=temporal_bilinear)
        voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)

    return voxel_grid
