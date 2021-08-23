import argparse
import cv2
import glob
import itertools
import os
from os import listdir
import pathlib

from matplotlib import pyplot as plt
import numpy as np

from PIL import Image as PILImage
from IPython.display import Image

from scipy.special import comb
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from scipy.stats import mode
from skimage.feature import register_translation


def simple_resize(img, long_side_out):
    """
    Resizes the image using a resize factor

    Args:
    img (numpy array): frame to be resized
    long_size_out (int): dimension which determines the resize factor

    Returns:
    resized_img (numpy array): resized frame

    """

    long_side = max(img.shape[:2])
    resize_factor = long_side_out/long_side
    resized_img = cv2.resize(img, (int(resize_factor*img.shape[1]),int(resize_factor*img.shape[0])))

    return resized_img


def imgreg_dist(frame1, frame2):
    """
    Obtains the frame distance based on image registration with phase correlation (fast) + MAD

    Args:
    frame1 (numpy array): first frame in the pair to be processed
    frame2 (numpy array): second frame in the pair to be process

    Returns:
    MAD (float): mean absolute distance between frames
    RMSE_reg (float): registration error based on image registration with phase correlation
    
    """
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame1 = simple_resize(frame1, 480)
    frame2 = simple_resize(frame2, 480)
    
    # pad frames to prevent possible issues in FT space
    brd = 10
    frame1 = cv2.copyMakeBorder(frame1.copy(),brd,brd,brd,brd,cv2.BORDER_CONSTANT,value=[125])
    frame2 = cv2.copyMakeBorder(frame2.copy(),brd,brd,brd,brd,cv2.BORDER_CONSTANT,value=[125])
    
    # MAD error between initial frames
    MAD = np.mean(cv2.absdiff(frame1, frame2))
    # perform registration and obtain RMSE error between registered frames
    _, RMSE_reg, _ = register_translation(frame1, frame2)

    return MAD, RMSE_reg


def sample_video(video, frame_order, video_dir='forward', num_frames=10, sample_rate=6):
    """
    Samples frames from a video and stack them together

    Args:
    video (list): input video
    frame_order (list): ordered list of frames

    Returns:
    collage (numpy array): sample video frames

    """
    if video_dir == 'forward':
        frame_order=frame_order
    if video_dir == 'reverse':
        frame_order=frame_order[::-1]
    frame_order=frame_order[0:sample_rate*num_frames:sample_rate]
    
    collage = video[frame_order[0]]
    for f in frame_order[1:]:
        collage = np.concatenate((collage, video[f]), axis=0)

    return collage


def frames_similarity(video):
    """
    Compute frames similarity using MAD and RMSE

    Args:
    video (list): input video

    Returns:
    dist_mad_list (list): list of mean absolute distance values
    dist_m (numpy array): distance matrix

    """
    # compute distances between all frame pairs
    num_frames = len(video)
    num_pairs = comb(num_frames, 2, exact=True)
    img_reg_dist_list = []
    c=1
    for subset in itertools.combinations(video, 2):
        img_reg_dist_list.append(imgreg_dist(subset[0],subset[1]))
        print('Computing distance for a pair {}/{}'.format(c, num_pairs))
        c+=1

    # MAD image distances for outliers detection
    max_dist_val = max(np.asarray(img_reg_dist_list)[:,0])
    dist_mad_list = [x[0]/max_dist_val for x in img_reg_dist_list]

    # RMSE registration based distances for frame ordering
    dist_reg_list = [x[1] for x in img_reg_dist_list]
    # convert pair list into a dissimilarity/distance matrix
    dist_m = squareform(dist_reg_list)

    return dist_mad_list, dist_m


def clean_outliers(video, dist_mad_list, dist_m):
    """
    Compute frames similarity using MAD and RMSE

    Args:
    video (list): input video
    dist_mad_list (list): list of mean absolute distance values
    dist_m (numpy array): distance matrix

    Returns:
    video_clean (list): input video without outlier frames
    dist_m (numpy array): distance matrix without outliers

    """
    # Perform hierarchical clustering and show a tree dendrogram (using coarse image distance)
    linkage_matrix = hierarchy.linkage(dist_mad_list, 'single')
    fig = plt.figure(figsize=(25, 10))
    dn = hierarchy.dendrogram(linkage_matrix,color_threshold=0.15)
    plt.show()

    # !! Careful: Execute only if outliers are present in the corrupted video
    # Perform a tree cut and keep only the largest cluster of valid frames
    clust_cut=0.15
    clustering = hierarchy.fcluster(linkage_matrix, clust_cut, criterion='distance')
    good_frames = [clustering == mode(clustering)[0][0]][0]
    outliers = np.invert(good_frames)

    # clean video frames
    video_clean = [frame for (frame, keep) in zip(video, good_frames) if keep]

    # remove outliers rows/columns from distance matrix
    dist_m = dist_m[good_frames,:]
    dist_m = dist_m[:,good_frames]

    # save outlier frames
    video_outliers = [frame for (frame, keep) in zip(video, outliers) if keep]
    # clean output folder and write frames
    for file in glob.glob(output_dir_frames_outliers + '*.jpg'):
        os.remove(file)
    for idx,frame in enumerate(video_outliers):
        cv2.imwrite('{}/frame_{}.jpg'.format(output_dir_frames_outliers, idx), frame)
    print('Number of outlier frames: {}'.format(sum(outliers)))

    return video_clean, dist_m


def pivot_frame(dist_m, frame_order, frame_left, frame_right):
    """
    Obtain the order of the frames using pivot frames

    Args:
    dist_m (numpy array): distance matrix
    frame_order (list): ordered list of frames
    frame_left (list): selected pivot frame on the left
    frame_right (list): selected pivot frame on the right

    Returns:
    frame_order (list): updates ordered list of frames

    """

    # add frames to the left or to the right until all frames are in the list
    count = 2
    while count != dist_m.shape[0]:
        # find candidate frames closest to the current left/right frames
        frame_new_left = np.argmin(dist_m[frame_left,:])
        left_dist = dist_m[frame_left,frame_new_left]
        frame_new_right = np.argmin(dist_m[frame_right,:])
        right_dist = dist_m[frame_right,frame_new_right]
        if left_dist<right_dist:
            # add new frame on the left
            frame_order.insert(0,frame_new_left)
            dist_m[:,frame_left]=np.Inf
            dist_m[frame_left,:]=np.Inf
            frame_left = frame_new_left
        else:
            # add new frame on the right
            frame_order.append(frame_new_right)
            dist_m[:,frame_right]=np.Inf
            dist_m[frame_right,:]=np.Inf
            frame_right = frame_new_right
        count +=1
        # print(frame_order) # print frame aggregation process
    
    return frame_order


def order_frame(dist_m):
    """
    Order frames and display sample frames to dermine the direction of the video

    Args:
    dist_m (numpy array): distance matrix

    Returns:
    video_output (list): video containing the ordered frames

    """
    # put infinity on the distance matrix diagonal to avoid matchings with a frame itself
    np.fill_diagonal(dist_m, np.inf)

    # start with two frames having the minimal distance in the distance matrix
    frame_order = []
    idx = np.unravel_index(np.argmin(dist_m, axis=None), dist_m.shape)
    frame_left = idx[0]
    frame_right = idx[1]
    frame_order.extend([frame_left,frame_right])
    dist_m[frame_left,frame_right]=np.Inf
    dist_m[frame_right,frame_left]=np.Inf

    frame_order = pivot_frame(dist_m, frame_order, frame_left, frame_right)
    
    # show sampled frames from the beginning and the end
    video_sample_forward = sample_video(video_clean,frame_order,video_dir='forward')
    video_sample_reverse = sample_video(video_clean,frame_order,video_dir='reverse')

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(25, 40))
    ax1.axis("off")
    ax1.set_title('Order 1')
    ax1.imshow(cv2.cvtColor(video_sample_forward, cv2.COLOR_BGR2RGB))
    ax2.axis("off")
    ax2.set_title('Order 2')
    ax2.imshow(cv2.cvtColor(video_sample_reverse, cv2.COLOR_BGR2RGB))

    order_selection = input('Which sequence appears to be the start of the video (1/2)?\n')

    # reverse sequence if needed
    if order_selection=='1':
        frame_order=frame_order
    elif order_selection=='2':
        frame_order=frame_order[::-1]
    # order frames
    video_output = [video_clean[i] for i in frame_order]

    return video_output

        
parser = argparse.ArgumentParser(description='Handle model and dataset args.')
parser.add_argument('--input-video', type=str, default='shuffled_19.mp4', help='Path to input video')
parser.add_argument('--output-dir', type=str, default='output', help='Path to output directory')


if __name__ == '__main__':
    args = parser.parse_args()

    output_dir_frames_orig = os.path.join(args.output_dir, 'frames_original')
    pathlib.Path(output_dir_frames_orig).mkdir(parents=True, exist_ok=True)
    output_dir_frames_corr = os.path.join(args.output_dir, 'frames_corrected')
    pathlib.Path(output_dir_frames_corr).mkdir(parents=True, exist_ok=True)
    output_dir_frames_outliers = os.path.join(args.output_dir, 'frames_corrected', 'outliers')
    pathlib.Path(output_dir_frames_outliers).mkdir(parents=True, exist_ok=True)
    output_video_corr = os.path.join(args.output_dir, 'video_corrected.mp4')

    vidcap = cv2.VideoCapture(args.input_video)
    reading_flag,image = vidcap.read()
    video = []
    count=0

    while reading_flag:
        video.append(image)     
        cv2.imwrite('{}/frame_{}.jpg'.format(output_dir_frames_orig, count), image)     # save original frames   
        count += 1
        reading_flag,image = vidcap.read()

    dist_mad_list, dist_m = frames_similarity(video)
    video_clean, dist_m = clean_outliers(video, dist_mad_list, dist_m)
    video_output = order_frame(dist_m)

    # clean output folder
    for file in glob.glob(output_dir_frames_corr + '*.jpg'):
        os.remove(file)
    # save frames as images
    for idx,frame in enumerate(video_output):
        cv2.imwrite('{}/frame_{}.jpg'.format(output_dir_frames_corr, idx), frame)

    # save video
    out = cv2.VideoWriter(output_video_corr,cv2.VideoWriter_fourcc(*'mp4v'), 24, (video_output[0].shape[1],video_output[0].shape[0]))
    for frame in video_output:
        frame = cv2.resize(frame,(frame.shape[1],frame.shape[0]))
        out.write(frame)
    out.release()
    print("Video saved as {}".format(output_video_corr))
    