import os
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import csv
import cv2

# Function to merge optical flow components into RGB representation
def merge_optical_flow(flow):
    
    flow_x, flow_y = flow[0,...], flow[1,...]
    h, w = flow_x.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = 255
    hsv[..., 1] = 255

    # Convert flow to polar coordinates (magnitude, angle)
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)

    # Map angle to hue
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Normalize magnitude to 0-255
    # hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = (255 * magnitude / magnitude.max()).astype(np.uint8)
    # Convert HSV to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # rgb = cv2.applyColorMap(rgb, cv2.COLORMAP_PARULA) #COLORMAP_PARULA cv2.COLORMAP_SPRING cv2.COLORMAP_COOL
    return rgb


def make_event_preview(events, mode='grayscale', num_bins_to_show=-1):
    # events: [1 x C x H x W] event numpy or [C x H x W]
    # num_bins_to_show: number of bins of the voxel grid to show. -1 means show all bins.
    if events.ndim == 3:
        events = np.expand_dims(events,axis=0)
    if num_bins_to_show < 0:
        sum_events = np.sum(events[0, :, :, :], axis=0)
        # sum_events = np.sum(events[0, :, :, :], axis=0)
    else:
        sum_events = np.sum(events[0, -num_bins_to_show:, :, :], axis=0)

    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.zeros((sum_events.shape[0], sum_events.shape[1], 3), dtype=np.uint8)
        b = event_preview[:, :, 0]
        r = event_preview[:, :, 2]
        b[sum_events > 0] = 255
        r[sum_events < 0] = 255
    else:
        # Grayscale mode
        # normalize event image to [0, 255] for display
        m, M = -5.0, 5.0
        # M = (sum_events.max() - sum_events.min())/2
        # m = -M
        
        event_preview = np.clip((255.0 * (sum_events - m) / (M - m)), 0, 255).astype(np.uint8)
        # event_preview = np.clip((255.0 * (sum_events - sum_events.min()) / (sum_events.max() - sum_events.min())).astype(np.uint8), 0, 255)

    return event_preview


class Writer:
    def __init__(self, cfgs, model_name, dataset_name=None):
        self.output_folder = cfgs.output_folder
        if not dataset_name:
            self.dataset_name = cfgs.test_data_name
        else:
            self.dataset_name = dataset_name

        if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
        self.output_data_folder = os.path.join(self.output_folder, model_name, '{}'.format(self.dataset_name))


class EvalWriter(Writer):
    """
    Write evaluation results to disk.
    """

    def __init__(self, cfgs, model_name, dataset_name=None):
        super(EvalWriter, self).__init__(cfgs, model_name, dataset_name)
        self.is_write_image = cfgs.is_write_image
        print('== Eval Txt Writer ==')
        if self.is_write_image:
            if not os.path.exists(self.output_data_folder):
                os.makedirs(self.output_data_folder)
            self.output_txt_file = os.path.join(self.output_data_folder, 'result.csv')
            print('Will write evaluation result to: {}'.format(self.output_txt_file))
        else:
            print('Will not write evaluation result to disk.')

    def __call__(self, name_results, results):
        if not self.is_write_image:
            return
        with open(self.output_txt_file, 'a+', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            # mse, psnr, ssim, lpips, num_frame, N_events
            writer.writerow(name_results)
            writer.writerow(results)
        f.close()


class ErrorMapWriter(Writer):
    """
    Write error_map between reconstructed and GT images to disk.
    """

    def __init__(self, cfgs, model_name, dataset_name=None):
        super(ErrorMapWriter, self).__init__(cfgs, model_name, dataset_name)
        self.is_write_emap = cfgs.is_write_emap
        
        if not dataset_name:
            self.dataset_name = cfgs.test_data_name
        else:
            self.dataset_name = dataset_name

        print('== Error Map Writer ==')
        if self.is_write_emap:
            self.output_data_folder = os.path.join(self.output_data_folder, 'error_maps')
            if not os.path.exists(self.output_data_folder):
                os.makedirs(self.output_data_folder)
            print('Will write error maps to: {}'.format(self.output_data_folder))
        else:
            print('Will not write error maps to disk.')

    def __call__(self, img, gt_img, img_id):
        if not self.is_write_emap:
            return
        diff = img.astype(np.float32)/255.- gt_img.astype(np.float32)/255.
    
        plt.imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)  #coolwarm
        plt.axis('off') 
        plt.savefig(os.path.join(self.output_data_folder,
                         'frame_{:010d}.png'.format(img_id)), bbox_inches='tight')


class ImageWriter(Writer):
    """
    Utility class to write images to disk.
    """

    def __init__(self, cfgs, model_name, dataset_name=None):
        super(ImageWriter, self).__init__(cfgs, model_name, dataset_name)
        self.is_write_image = cfgs.is_write_image
        
        print('== Image Writer ==')
        if self.is_write_image:
            if not os.path.exists(self.output_data_folder):
                os.makedirs(self.output_data_folder)
            print('Will write images to: {}'.format(self.output_data_folder))
        else:
            print('Will not write images to disk.')

    def __call__(self, img, img_id):
        if not self.is_write_image:
            return
        img = Image.fromarray(np.uint8(img))
        img.save(os.path.join(self.output_data_folder,
                         'frame_{:010d}.png'.format(img_id)))


class FlowWriter(Writer):
    """
    Utility class to write event images to disk.
    """

    def __init__(self, cfgs, model_name, dataset_name=None):
        super(FlowWriter, self).__init__(cfgs, model_name, dataset_name)

        self.is_write_flow = cfgs.is_write_flow

        print('== Event Writer ==')
        if self.is_write_flow:
            self.output_data_folder = os.path.join(self.output_data_folder, 'flow')
            if not os.path.exists(self.output_data_folder):
                os.makedirs(self.output_data_folder)
            print('Will write flow to: {}'.format(self.output_data_folder))
        else:
            print('Will not write flow to disk.')
    
    
    def __call__(self, flow, img_id):
        if not self.is_write_flow:
            return
        rgb_frame = merge_optical_flow(flow)

        # Save RGB frame
        cv2.imwrite(os.path.join(self.output_data_folder,
                         'flow_{:010d}.png'.format(img_id)), 
                    rgb_frame)


class EventWriter(Writer):
    """
    Utility class to write event images to disk.
    """

    def __init__(self, cfgs, model_name, dataset_name=None, save_folder_name='events'):
        super(EventWriter, self).__init__(cfgs, model_name, dataset_name)

        self.is_write_event = cfgs.is_write_event

        print('== Event Writer ==')
        if self.is_write_event:
            self.output_data_folder = os.path.join(self.output_data_folder, save_folder_name) #'events'
            if not os.path.exists(self.output_data_folder):
                os.makedirs(self.output_data_folder)
            print('Will write event tensor to: {}'.format(self.output_data_folder))
        else:
            print('Will not write events to disk.')

    def __call__(self, img, img_id):
        if not self.is_write_event:
            return
        img = Image.fromarray(np.uint8(img))
        img.save(os.path.join(self.output_data_folder,
                         'events_{:010d}.png'.format(img_id)))


### NOT WORK
class VideoWriter(Writer):
    def __init__(self, cfgs, model_name, video_name, dataset_name=None):
        super(VideoWriter, self).__init__(cfgs, model_name, dataset_name)

        self.is_write_video = cfgs.is_write_video

        print('== Video Writer ==')
        if self.is_write_video:
            self.output_data_folder = os.path.join(self.output_data_folder, 'video')
            fps = 12
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video = cv2.VideoWriter(os.path.join(self.output_data_folder, video_name+'.mp4'),fourcc, fps, (cfgs.image_dim[1],cfgs.image_dim[0]))
            self.img_list = []
            if not os.path.exists(self.output_data_folder):
                os.makedirs(self.output_data_folder)
            print('Will write images as a video to: {}'.format(self.output_data_folder))
        else:
            print('Will not write video to disk.')
    
    # def __del__(self):
    #     # cv2.destroyAllWindows()
    #     # self.video.release()
    #     if self.is_write_video:
    #         self.video.release()
    
    def write_video(self):
        fps = 12
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = self.img_list[0].shape[-2:]
        self.video = cv2.VideoWriter(os.path.join(self.output_data_folder, video_name+'.mp4'),fourcc, fps, (width,height))
        for img in self.img_list:
            self.video.write(img)
        self.video.release()
        cv2.destroyAllWindows()
        
    def __call__(self, img):
        if not self.is_write_video:
            return
        
        if img.dtype == np.float32:
            img = np.uint8(255.0*img)
        self.img_list.append(img)
        
    

def show_whole_img(event_patch, output, gt_img_patch):
    plt.subplot(1,3,1)
    events = event_patch.cpu().data #patch_maker.merge_patches_to_image_tensor(event_patch[:,-1,:,:].cpu().data)
    plt.imshow(events[0,0,:,:].numpy())#, cmap='gray')
    # plt.title('Event')
    plt.title('mean {:.3f}, var {:.3f}'.format(event_patch.mean(), event_patch.var()))
    plt.subplot(1,3,2)
    # pred_img = patch_maker.merge_patches_to_image_tensor(output.cpu().data)
    pred_img = output.cpu().data
    plt.imshow(pred_img[0,0,:,:].numpy(), cmap='gray')
    # plt.title('Pred')
    plt.title('mean {:.3f}, var {:.3f}'.format(pred_img.mean(), pred_img.var()))
    #plt.axis('off')
    plt.subplot(1,3,3)
    gt_img = gt_img_patch.cpu().data
    plt.imshow(gt_img[0,0,:,:])#, cmap='gray')
    # plt.title('Ground Truth') #Ground Truth
    plt.title('mean {:.3f}, var {:.3f}'.format(gt_img.mean(), gt_img.var()))
    #plt.axis('off')
    # plt.show()
    plt.savefig('test_images/train_rec.png')


def show_flow(event_patch, pred_flow, gt_flow, warped_pred_I, warped_gt_I):
    row = 2
    # plt.subplot(row,2,1)
    # events = event_patch.cpu().data #patch_maker.merge_patches_to_image_tensor(event_patch[:,-1,:,:].cpu().data)
    # plt.imshow(events[0,0,:,:].numpy(), cmap='gray')
    # plt.title('Event')
    # plt.axis('off')
    # plt.subplot(row,2,2)
    # gt_flow = gt_flow.cpu().data
    # plt.imshow(gt_flow[0,0,:,:].numpy()) #, cmap='gray'
    # plt.title('GT flow')
    # plt.axis('off')
    plt.subplot(row,2,1)
    gt_flow = gt_flow.cpu().data.numpy()
    plt.imshow(gt_flow[0,0,:,:]) #, cmap='gray'
    plt.title('GT flow') #Ground Truth
    plt.axis('off')
    plt.subplot(row,2,2)
    pred_flow = pred_flow.cpu().data.numpy()
    plt.imshow(pred_flow[0,0,:,:])
    plt.title('Pred flow') #Ground Truth
    plt.axis('off')
    plt.subplot(row,2,3)
    warped_gt_I = warped_gt_I.cpu().data.numpy()
    plt.imshow(warped_gt_I[0,0,:,:], cmap='gray')
    plt.title('Diff w gt flow') #Ground Truth
    plt.axis('off')
    plt.subplot(row,2,4)
    warped_pred_I = warped_pred_I.cpu().data.numpy()
    plt.imshow(warped_pred_I[0,0,:,:], cmap='gray')
    plt.title('Diff w pred flow') #Ground Truth
    plt.axis('off')
    
    # plt.show()
    plt.savefig('test_images/flow.png')
