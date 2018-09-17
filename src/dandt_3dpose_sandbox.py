import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import data_utils
import viz
import re
import cameras
import json
import os
import time
from predict_3dpose import create_model
import cv2
import imageio
import logging
import scipy as sp
from pprint import pprint
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

FLAGS = tf.app.flags.FLAGS

#order = [15, 12, 25, 26, 27, 17, 18, 19, 1, 2, 3, 6, 7, 8]

order = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def show_anim_curves(anim_dict, _plt):
    val = np.array(list(anim_dict.values()))
    for o in range(0,36,2):
        x = val[:,o]
        y = val[:,o+1]
        _plt.plot(x, 'r--', linewidth=0.2)
        _plt.plot(y, 'g', linewidth=0.2)
    return _plt

openpose_output_dir = FLAGS.openpose
    
level = {0:logging.ERROR,
         1:logging.WARNING,
         2:logging.INFO,
         3:logging.DEBUG}

logger.setLevel(level[FLAGS.verbose])


tf.app.run()



keypoints = np.load('/home/ubuntu/docker_volumes/data2/pullups_keypoints.npy')    
boxes = np.load('/home/ubuntu/docker_volumes/data2/pullups_boxes.npy')

new_keypoints = []

for frame in range(len(keypoints)):
    # Normalize x values
    keypoints[frame][0] = keypoints[frame][0] - boxes[frame][0]
    # Normalize y values
    keypoints[frame][0] = keypoints[frame][1] - boxes[frame][1]
    hip_x = (keypoints[frame][0][11] + keypoints[frame][0][12])/2
    hip_y = (keypoints[frame][1][11] + keypoints[frame][1][12])/2
    spine_x = keypoints[frame][0][1]
    spine_y = keypoints[frame][1][1]
    neck_nose_x = keypoints[frame][0][0]
    neck_nose_y = keypoints[frame][1][0]
    thorax_x = 2*keypoints[frame][0][2] - keypoints[frame][0][1]
    thorax_y = 2*keypoints[frame][1][2] - keypoints[frame][1][1]
    new_list = [
        hip_x, # 0 - hip - 11&12
	hip_y,
	keypoints[frame][0][12], # 1 - Right hip - 12
	keypoints[frame][1][12],
	keypoints[frame][0][14], # 2 - Righ knee - 14
	keypoints[frame][1][14],
	keypoints[frame][0][16], # 3 - Right foot - 16
        keypoints[frame][1][16],
	keypoints[frame][0][11], # 4 - Left hip - 11
        keypoints[frame][1][11],
	keypoints[frame][0][13], # 5 - Left knee - 13
        keypoints[frame][1][13],
	keypoints[frame][0][15], # 6 - Left foot - 15
        keypoints[frame][1][15],
	spine_x, # 7 - Neck - 1
        spine_y,
	thorax_x, # 8 - Thorax - 2&1
	thorax_y,
	neck_nose_x, # 9 - Neck/Nose - Nose 0
	neck_nose_y,
	keypoints[frame][0][2], # 10 - Head - Top of head 2
	keypoints[frame][1][2],
	keypoints[frame][0][5], # 11 - Left shoulder - 5
	keypoints[frame][1][5], 
	keypoints[frame][0][7], # 12 - Left elbow - 7
        keypoints[frame][1][7],
	keypoints[frame][0][9], # 13 - Left wrist - 9
        keypoints[frame][1][9],
	keypoints[frame][0][6], # 14 - Right shoulder - 6
        keypoints[frame][1][6],
	keypoints[frame][0][8], # 15 - Right elbow - 8
        keypoints[frame][1][8],
	keypoints[frame][0][10], # 16 - Right wrist - 10
        keypoints[frame][1][10]
	]
    enc_in = np.zeros((1, 64))
    for i in range(len(order)):
	new_index = order[i]
	old_index = i
	for j in [0,1]:
	    enc_in[2*new_index + j] = keypoints[2*old_index + j]
    new_keypoints[frame] = enc_in

actions = data_utils.define_actions(FLAGS.action)

SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)

train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, FLAGS.data_dir)

train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14)

device_count = {"GPU": 1}
png_lib = []
    
with tf.Session(config=tf.ConfigProto(
            device_count=device_count,
            allow_soft_placement=True)) as sess:
    #plt.figure(3)
    batch_size = 128
    model = create_model(sess, actions, batch_size)
    iter_range = range(len(new_keypoints))
    for frame in iter_range:
	logger.info("calc frame {}/{}".format(frame, iter_range))
	enc_in = new_keypoints[frame][:,dim_to_use_2d]
	mu = data_mean_2d[dim_to_use_2d]
	stddev = data_std_2d[dim_to_use_2d]
	enc_in = np.divide((enc_in - mu), stddev)

	dp = 1.0
	dec_out = np.zeros((1, 48))
	dec_out[0] = [0 for i in range(48)]
	_, _, poses3d = model.step(sess, enc_in, dec_out, dp, isTraining=False)
	all_poses_3d = []
	enc_in = data_utils.unNormalizeData(enc_in, data_mean_2d, data_std_2d, dim_to_ignore_2d)
	poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)
	gs1 = gridspec.GridSpec(1, 1)
	gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
	plt.axis('off')
	all_poses_3d.append( poses3d )
	enc_in, poses3d = map( np.vstack, [enc_in, all_poses_3d] )
	subplot_idx, exidx = 1, 1
	_max = 0
	_min = 10000
	
	for i in range(poses3d.shape[0]):
            for j in range(32):
                tmp = poses3d[i][j * 3 + 2]
                poses3d[i][j * 3 + 2] = poses3d[i][j * 3 + 1]
                poses3d[i][j * 3 + 1] = tmp
                if poses3d[i][j * 3 + 2] > _max:
                    _max = poses3d[i][j * 3 + 2]
                if poses3d[i][j * 3 + 2] < _min:
                    _min = poses3d[i][j * 3 + 2]
	    
	    for i in range(poses3d.shape[0]):
                for j in range(32):
                    poses3d[i][j * 3 + 2] = _max - poses3d[i][j * 3 + 2] + _min
                    poses3d[i][j * 3] += (spine_x - 630)
                    poses3d[i][j * 3 + 2] += (500 - spine_y)
	    
	    # Plot 3d predictions
            ax = plt.subplot(gs1[subplot_idx - 1], projection='3d')
            ax.view_init(18, -70)    
            logger.debug(np.min(poses3d))
            if np.min(poses3d) < -1000:
                poses3d = before_pose

	    p3d = poses3d
            logger.debug(poses3d)
            viz.show3Dpose(p3d, ax, lcolor="#9b59b6", rcolor="#2ecc71")

            pngName = 'png/pose_frame_{0}.png'.format(str(frame).zfill(12))
            plt.savefig(pngName)
            if FLAGS.write_gif:
                png_lib.append(imageio.imread(pngName))
            before_pose = poses3d

if FLAGS.write_gif:
    if FLAGS.interpolation:
        #take every frame on gif_fps * multiplier_inv
        png_lib = np.array([png_lib[png_image] for png_image in range(0,len(png_lib), int(multiplier_inv)) ])
    logger.info("creating Gif png/animation.gif, please Wait!")
    imageio.mimsave('gif_output/animation.gif', png_lib, fps=FLAGS.gif_fps)

logger.info("Done!".format(pngName))

















	
	



enc_in = np.zeros((1, 64))
enc_in[0] = [0 for i in range(64)]

actions = data_utils.define_actions(FLAGS.action)

    SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
    rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(
        actions, FLAGS.data_dir)
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
        actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14)

    device_count = {"GPU": 1}
    png_lib = []
    with tf.Session(config=tf.ConfigProto(
            device_count=device_count,
            allow_soft_placement=True)) as sess:
        #plt.figure(3)
        batch_size = 128
        model = create_model(sess, actions, batch_size)
        iter_range = len(smoothed.keys())
        for n, (frame, xy) in enumerate(smoothed.items()):
            logger.info("calc frame {0}/{1}".format(frame, iter_range))
            # map list into np array  
            joints_array = np.zeros((1, 36))
            joints_array[0] = [0 for i in range(36)]
            for o in range(len(joints_array[0])):
                #feed array with xy array
                joints_array[0][o] = xy[o]
            _data = joints_array[0]
            # mapping all body parts or 3d-pose-baseline format
            for i in range(len(order)):
                for j in range(2):
                    # create encoder input
                    enc_in[0][order[i] * 2 + j] = _data[i * 2 + j]
            for j in range(2):
                # Hip
                enc_in[0][0 * 2 + j] = (enc_in[0][1 * 2 + j] + enc_in[0][6 * 2 + j]) / 2
                # Neck/Nose
                enc_in[0][14 * 2 + j] = (enc_in[0][15 * 2 + j] + enc_in[0][12 * 2 + j]) / 2
                # Thorax
                enc_in[0][13 * 2 + j] = 2 * enc_in[0][12 * 2 + j] - enc_in[0][14 * 2 + j]

            # set spine
            spine_x = enc_in[0][24]
            spine_y = enc_in[0][25]

            enc_in = enc_in[:, dim_to_use_2d]
            mu = data_mean_2d[dim_to_use_2d]
            stddev = data_std_2d[dim_to_use_2d]
            enc_in = np.divide((enc_in - mu), stddev)

            dp = 1.0
            dec_out = np.zeros((1, 48))
            dec_out[0] = [0 for i in range(48)]
            _, _, poses3d = model.step(sess, enc_in, dec_out, dp, isTraining=False)
            all_poses_3d = []
            enc_in = data_utils.unNormalizeData(enc_in, data_mean_2d, data_std_2d, dim_to_ignore_2d)
            poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)
            gs1 = gridspec.GridSpec(1, 1)
            gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
            plt.axis('off')
            all_poses_3d.append( poses3d )
            enc_in, poses3d = map( np.vstack, [enc_in, all_poses_3d] )
            subplot_idx, exidx = 1, 1
            _max = 0
            _min = 10000

            for i in range(poses3d.shape[0]):
                for j in range(32):
                    tmp = poses3d[i][j * 3 + 2]
                    poses3d[i][j * 3 + 2] = poses3d[i][j * 3 + 1]
                    poses3d[i][j * 3 + 1] = tmp
                    if poses3d[i][j * 3 + 2] > _max:
                        _max = poses3d[i][j * 3 + 2]
                    if poses3d[i][j * 3 + 2] < _min:
                        _min = poses3d[i][j * 3 + 2]

            for i in range(poses3d.shape[0]):
                for j in range(32):
                    poses3d[i][j * 3 + 2] = _max - poses3d[i][j * 3 + 2] + _min
                    poses3d[i][j * 3] += (spine_x - 630)
                    poses3d[i][j * 3 + 2] += (500 - spine_y)

            # Plot 3d predictions
            ax = plt.subplot(gs1[subplot_idx - 1], projection='3d')
            ax.view_init(18, -70)    
            logger.debug(np.min(poses3d))
            if np.min(poses3d) < -1000:
                poses3d = before_pose

            p3d = poses3d
            logger.debug(poses3d)
            viz.show3Dpose(p3d, ax, lcolor="#9b59b6", rcolor="#2ecc71")

            pngName = 'png/pose_frame_{0}.png'.format(str(frame).zfill(12))
            plt.savefig(pngName)
            if FLAGS.write_gif:
                png_lib.append(imageio.imread(pngName))
            before_pose = poses3d

    if FLAGS.write_gif:
        if FLAGS.interpolation:
            #take every frame on gif_fps * multiplier_inv
            png_lib = np.array([png_lib[png_image] for png_image in range(0,len(png_lib), int(multiplier_inv)) ])
        logger.info("creating Gif png/animation.gif, please Wait!")
        imageio.mimsave('gif_output/animation.gif', png_lib, fps=FLAGS.gif_fps)
    logger.info("Done!".format(pngName))


if __name__ == "__main__":

    openpose_output_dir = FLAGS.openpose
    
    level = {0:logging.ERROR,
             1:logging.WARNING,
             2:logging.INFO,
             3:logging.DEBUG}

    logger.setLevel(level[FLAGS.verbose])


    tf.app.run()
