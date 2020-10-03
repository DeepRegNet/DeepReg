import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse
import tensorflow as tf

os.chdir(r'/home/s-sd/Desktop/deepreg_project/DeepReg')
from deepreg.dataset.loader.nifti_loader import load_nifti_file
from deepreg.model.layer_util import warp_image_ddf


def string_to_list(string):
    '''
    converts a comma separated string to a list of strings
    also removes leading or trailing spaces from each element in list
    '''
    return [elem.strip() for elem in string.split(',')]

def gif_slices(img_paths, save_path="", interval=50):
    if type(img_paths) is str:
        img_paths = string_to_list(img_paths)
    img = load_nifti_file(img_paths[0])
    img_shape = np.shape(img)
    for img_path in img_paths:
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        img = load_nifti_file(img_path)
    
        frames = []
        for index in range(img_shape[-1]):
            frame = plt.imshow(img[:, :, index], aspect='auto', animated=True)
            # plt.axis('off')
            frames.append([frame])
        
        anim = animation.ArtistAnimation(fig, frames, interval=interval)


        path_to_anim_save = os.path.join(save_path, os.path.split(img_path)[-1].split('.')[0]+'.gif')
        
        anim.save(path_to_anim_save)
        print("Animation saved to:", path_to_anim_save)

    
def tile_slices(img_paths, save_path="", fname=None, slice_inds=None, col_titles=None):
    if type(img_paths) is str:
        img_paths = string_to_list(img_paths)
    img = load_nifti_file(img_paths[0])
    img_shape = np.shape(img)
    
    if slice_inds is None:
        slice_inds = [round(np.random.rand() * (img_shape[-1])-1)]
    
    if col_titles is None:
        col_titles = [os.path.split(img_path)[-1].split('.')[0] for img_path in img_paths]
    
    num_inds = len(slice_inds)
    num_imgs = len(img_paths)
    
    subplot_mat = np.array(np.arange(num_inds * num_imgs) +1).reshape(num_inds, num_imgs)
    
    plt.figure(figsize=(num_imgs*2, num_inds*2))
    
    imgs = []
    for img_path in img_paths:
        img = load_nifti_file(img_path)
        imgs.append(img)
    
    for img, col_num in zip(imgs, range(num_imgs)):
        for index, row_num in zip(slice_inds, range(num_inds)):
            plt.subplot(num_inds, num_imgs, subplot_mat[row_num, col_num])
            plt.imshow(img[:, :, index])
            plt.axis('off')
            if row_num-0<1e-3:
                plt.title(col_titles[col_num])
    
    if fname is None:
        fname = 'visualisation.png'
    save_fig_to = os.path.join(save_path, fname)
    plt.savefig(save_fig_to)
    print('Plot saved to:', save_fig_to)
    
    
    
def gif_warp(img_paths, ddf_path, slice_inds=None, num_interval=100, interval=50, save_path=""):
    if type(img_paths) is str:
        img_paths = string_to_list(img_paths)
    
    image = load_nifti_file(img_paths[0])
    img_shape = np.shape(image)
    
    if slice_inds is None:
        slice_inds = [round(np.random.rand() * (img_shape[-1])-1)]
    
    
    if slice_inds is None:
        np.random.rand()
    
    for img_path in img_paths:
        for slice_ind in slice_inds:
            
            fig = plt.figure()
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            
            ddf_scalers = np.linspace(0, 1, num=num_interval)
            
            frames = []
            for ddf_scaler in ddf_scalers:
                image = load_nifti_file(img_path)
                ddf = load_nifti_file(ddf_path)
                image = np.expand_dims(image, axis=0)
                ddf = np.expand_dims(ddf, axis=0) * ddf_scaler
            
                warped_image = warp_image_ddf(image=image, ddf=ddf, grid_ref=None)
                warped_image = np.squeeze(warped_image.numpy())
                
                frame = plt.imshow(warped_image[:, :, slice_ind], aspect='auto', animated=True)
                
                frames.append([frame])
                
            anim = animation.ArtistAnimation(fig, frames, interval=interval)
            path_to_anim_save = os.path.join(save_path, os.path.split(img_path)[-1].split('.')[0] + '_slice_' + str(slice_ind) +'.gif')
            
            anim.save(path_to_anim_save)
            print('Animation saved to:', path_to_anim_save)


        

def main(args=None):
    parser = argparse.ArgumentParser(
    description="deepreg_vis", formatter_class=argparse.RawTextHelpFormatter
    )


    parser.add_argument("--mode", "-m", help="Mode of visualisation \n0 for animtion over image slices, \n1 for warp animation, \n2 for tile plot",
        type=int, required=True)
    parser.add_argument("--image-paths", "-i", help="File path for image file (can specify multiple paths using a comma separated string)", type=str, required=True)
    parser.add_argument("--save-path", "-s", help="Path to directory where resulting visualisation is saved", default="")


    parser.add_argument("--interval", help="Interval between frames of animation (in miliseconds) \nApplicable only if --mode 0 or --mode 1", type=int, default=50)
    parser.add_argument("--ddf-path", help="Path to ddf used for warping image/s \nApplicable only and required if --mode 1", type=str, default=None)
    parser.add_argument("--num-interval", help="Number of intervals to use for warping \nApplicable only if --mode 1", type=int, default=100)
    parser.add_argument("--slice-inds", help="Comma separated string of indexes of slices to be used for the visualisation \nApplicable only if --mode 1 or --mode 2", type=str, default=None)
    parser.add_argument("--fname", help="File name (with extension like .png, .jpeg, ...) to save visualisation to \nApplicable only if --mode 2", type=str, default=None)
    parser.add_argument("--col-titles", help="Comma separated string of column titles to use (inferred from file names if not provided) \nApplicable only if --mode 2", default=None)


    # init arguments
    args = parser.parse_args(args)
    
    
    if args.slice_inds is not None:
        args.slice_inds = string_to_list(args.slice_inds)
        args.slice_inds = [int(elem) for elem in args.slice_inds]
    
    if args.mode is int(0):
        gif_slices(img_paths=args.image_paths, save_path=args.save_path, interval=args.interval)
    elif args.mode is int(1):
        if args.ddf_path is None:
            raise Exception('--ddf-path is required when using --mode 1')
        gif_warp(img_paths=args.image_paths, ddf_path=args.ddf_path, slice_inds=args.slice_inds, num_interval=args.num_interval, interval=args.interval, save_path=args.save_path)
    elif args.mode is int(2):
        tile_slices(img_paths=args.image_paths, save_path=args.save_path, fname=args.fname, slice_inds=args.slice_inds, col_titles=args.col_titles)

if __name__ == "__main__":
    main()