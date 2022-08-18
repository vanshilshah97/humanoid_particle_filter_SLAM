import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
import os
import sys
import time
import p3_util as ut
from read_data import LIDAR, JOINTS
import probs_utils as prob
import math
import cv2
import transformations as tf
from copy import deepcopy
import logging
from math import pi
import pdb
import matplotlib.pyplot as plt
import gen_figures as gf

if (sys.version_info > (3,0)):
    import pickle
else:
    import cPickle as pickle

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class SLAM(object):
    def __init__(self):
        self._characterize_sensor_specs()

    def _read_data(self, src_dir, dataset=0, split_name='train'):
        self.dataset_ = str(dataset)

        if split_name.lower() not in src_dir:
            src_dir = src_dir + '/' + split_name

        print('\n------Reading Lidar and Joints (IMU)------')
        lidar_data_ = split_name + '_lidar' + self.dataset_
        self.lidar_ = LIDAR(dataset=self.dataset_, data_folder=src_dir, name=lidar_data_)

        print('\n------Reading Joints Data------')
        joint_data_ = split_name + '_joint' + self.dataset_
        self.joints_ = JOINTS(dataset=self.dataset_, data_folder=src_dir, name=joint_data_)

        self.num_data_ = len(self.lidar_.data_)

        # Position of indices
        self.odo_indices_ = np.empty((2, self.num_data_), dtype=np.int64)

    def _characterize_sensor_specs(self, p_thresh=None):
        # Height of the lidar from the ground
        self.h_lidar_ = 0.93 + 0.33 + 0.15
        self.h_com_ = 0.93

        # Accuracy of the lidar
        self.p_true_ = 9
        self.p_false_ = 1.0/9

        #TODO set a threshold value of probability to consider a map's cell occupied
        self.p_thresh_ = 0.6 if p_thresh is None else p_thresh

        # Compute the corresponding threshold value of logodd
        self.logodd_thresh_ = prob.log_thresh_from_pdf_thresh(self.p_thresh_)
        self.ground_threshold_ = 0.2

    def _init_particles(self, num_p=0, mov_cov=None, particles=None, weights=None, percent_eff_p_thresh=None):
        self.num_p_ = num_p
        # Initialize particles
        self.particles_ = np.zeros((3, self.num_p_), dtype=np.float64) if particles is None else particles
        # Weights for the particles
        self.weights_ = 1.0/self.num_p_*np.ones(self.num_p_) if weights is None else weights

        # Position of the best particle after update on the map
        self.best_p_indices_ = np.zeros((2, self.num_data_), dtype=np.int64)
        # Best particles
        self.best_p_ = np.zeros((3, self.num_data_))
        # Corresponding time stamps of best particles
        self.time_ = np.zeros(self.num_data_)

        # Covariance matrix of motion model
        tiny_mov_cov = np.array([[1e-8, 0, 0],[0, 1e-8, 0],[0, 0, 1e-8]])
        self.mov_cov_ = mov_cov if mov_cov is not None else tiny_mov_cov

        # Threshold for resampling the particles
        self.percent_eff_p_thresh_ = percent_eff_p_thresh

    def _init_map(self, map_resolution=0.05):
        """ Input: resolution of map """
        MAP = {}
        MAP['res'] = map_resolution
        MAP['xmin'] = -20
        MAP['ymin'] = -20
        MAP['xmax'] = 20
        MAP['ymax'] = 20
        MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))
        MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

        MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)

        self.MAP_ = MAP

        self.log_odds_ = np.zeros((self.MAP_['sizex'], self.MAP_['sizey']), dtype=np.float64)
        self.occu_ = np.ones((self.MAP_['sizex'], self.MAP_['sizey']), dtype=np.uint64)
        # Number of measurements for each cell
        self.num_m_per_cell_ = np.zeros((self.MAP_['sizex'], self.MAP_['sizey']), dtype=np.uint64)

    def _build_first_map(self, t0=0, use_lidar_yaw=True):
        """ Build the first map using first lidar"""
        self.t0 = t0

        MAP = self.MAP_
        print('\n -------- Building the first map --------')

        joint_idx = np.argmin(abs(self.lidar_.data_[t0]['t'][0] - self.joints_.data_['ts'][0]))
        neck_angle, head_angle  = self.joints_.data_['head_angles'][:,joint_idx]

        R_pose = self.lidar_.data_[t0]['pose'][0]
        ray_l = self.lidar_.data_[t0]['scan'][0]

        ray_angles = np.linspace(-2.355, 2.355, len(ray_l))

        valid_id = np.logical_and((ray_l >= 0.1), (ray_l <= 30))
        ray_l = ray_l[valid_id]
        ray_angles = ray_angles[valid_id]

        lidar_to_body = self.lidar_._lidar_to_body_homo(head_angle, neck_angle)

        world_to_body_trans = np.array([R_pose[0], R_pose[1], self.h_com_])
        world_to_body_rot = tf.rot_z_axis(R_pose[2])

        world_to_body_homo = tf.homo_transform(world_to_body_rot, world_to_body_trans)

        x_val = ray_l * np.cos(ray_angles)
        y_val = ray_l * np.sin(ray_angles)
        z_val = np.zeros_like(x_val)
        ones = np.ones_like(x_val)

        points = np.vstack((x_val, y_val, z_val, ones))

        points_body = np.matmul(lidar_to_body, points)
        points_world = np.matmul(world_to_body_homo, points_body)

        points_world = points_world[:2,points_world[2,:] > self.ground_threshold_]

        map_cells = np.array(self.lidar_._physicPos2Pos(MAP, points_world))
        valid_map_id = np.logical_and(np.logical_and((map_cells[0] >= 0), (map_cells[0] < MAP['sizex'])), np.logical_and((map_cells[1] >= 0), map_cells[1] < MAP['sizey']))
        map_cells = map_cells[:,valid_map_id]

        self.log_odds_[map_cells[0], map_cells[1]] += (2 * np.log(9))

        R_pose_cells = self.lidar_._physicPos2Pos(MAP, R_pose[:2])
        self.best_p_[:, t0] = R_pose
        self.best_p_indices_[:, t0] = R_pose_cells

        x = np.append(map_cells[0], R_pose_cells[0])
        y = np.append(map_cells[1], R_pose_cells[1])
        contors = np.array([y, x]).T.astype(np.int)

        mask = np.zeros_like(self.log_odds_)
        cv2.drawContours(image=mask, contours=[contors], contourIdx=0, color=np.log(self.p_false_), thickness=cv2.FILLED)

        self.log_odds_ += mask

        MAP['map'] = 1*(self.log_odds_>self.logodd_thresh_)
        # End code
        self.MAP_ = MAP


    def _predict(self, t, use_lidar_yaw=True):
        logging.debug('\n -------- Doing prediction at t = {0} --------'.format(t))
        #TODO Integrate odometry later

        odom = tf.twoDSmartMinus(self.lidar_.data_[t]['pose'][0], self.lidar_.data_[t-1]['pose'][0])

        noise_vector = np.random.multivariate_normal([0,0,0], self.mov_cov_, self.num_p_).T


        for i in range(self.num_p_):
            self.particles_[:,i] = tf.twoDSmartPlus(tf.twoDSmartPlus(self.particles_[:,i],odom), noise_vector[:,i])
            # new_particles[:,i] = tf.twoDSmartPlus(self.particles_[:,i], noise_vectors[:,i])
            # new_particles[:,i] = tf.twoDSmartPlus(self.particles_[:,i], tf.twoDSmartMinus(odom_curr, odom_prev))


    def _update(self,t,t0=0,fig='on'):
        """Update function where we update the """
        if t == t0:
            self._build_first_map(t0,use_lidar_yaw=True)
            return
        # for one particle

        MAP = self.MAP_
        # joint_idx = np.where(self.lidar_.data_[t]['t'][0]<=self.joints_.data_['ts'][0])[0][0]
        joint_idx = np.argmin(abs(self.lidar_.data_[t]['t'][0] - self.joints_.data_['ts'][0]))
        neck_angle, head_angle = self.joints_.data_['head_angles'][:, joint_idx]

        ray_l = self.lidar_.data_[t]['scan'][0]

        ray_angles = np.linspace(-2.355, 2.355, len(ray_l))

        valid_id = np.logical_and((ray_l >= 0.1), (ray_l <= 30))
        ray_l = ray_l[valid_id]
        ray_angles = ray_angles[valid_id]

        lidar_to_body = self.lidar_._lidar_to_body_homo(head_angle, neck_angle)

        x_val = ray_l * np.cos(ray_angles)
        y_val = ray_l * np.sin(ray_angles)
        z_val = np.zeros_like(x_val)
        ones = np.ones_like(x_val)

        points = np.vstack((x_val, y_val, z_val, ones))

        points_body = np.matmul(lidar_to_body, points)

        correlations = np.zeros(self.num_p_)

        for i in range(self.num_p_):
            R_pose = self.particles_[:,i]

            world_to_body_trans = np.array([R_pose[0], R_pose[1], self.h_com_])
            world_to_body_rot = tf.rot_z_axis(R_pose[2])
            world_to_body_homo = tf.homo_transform(world_to_body_rot, world_to_body_trans)

            points_world = np.matmul(world_to_body_homo, points_body)

            points_world = points_world[:2,points_world[2,:] > self.ground_threshold_]

            map_cells = np.array(self.lidar_._physicPos2Pos(MAP, points_world))
            valid_map_id = np.logical_and(np.logical_and((map_cells[0] >= 0), (map_cells[0] < MAP['sizex'])),
                                          np.logical_and((map_cells[1] >= 0), map_cells[1] < MAP['sizey']))
            map_cells = map_cells[:, valid_map_id]

            correlations[i] = prob.mapCorrelation(MAP['map'], map_cells)

        self.weights_ = prob.update_weights(self.weights_, correlations)

        best_particle_id = np.argmax(self.weights_)
        R_pose = self.particles_[:,best_particle_id]

        #if t % 100 == 0:
        #    print(self.weights_)

        self.best_p_[:,t] = R_pose
        self.best_p_indices_[:,t] = self.lidar_._physicPos2Pos(MAP, R_pose[:2])

        world_to_body_trans = np.array([R_pose[0], R_pose[1], self.h_com_])
        world_to_body_rot = tf.rot_z_axis(R_pose[2])
        world_to_body_homo = tf.homo_transform(world_to_body_rot, world_to_body_trans)

        points_world = np.matmul(world_to_body_homo, points_body)

        points_world = points_world[:2, points_world[2, :] > self.ground_threshold_]

        map_cells = np.array(self.lidar_._physicPos2Pos(MAP, points_world))
        valid_map_id = np.logical_and(np.logical_and((map_cells[0] >= 0), (map_cells[0] < MAP['sizex'])),
                                      np.logical_and((map_cells[1] >= 0), map_cells[1] < MAP['sizey']))
        map_cells = map_cells[:, valid_map_id]

        self.log_odds_[map_cells[0], map_cells[1]] += (2 * np.log(9))

        R_pose_cells = self.best_p_indices_[:,t]

        x = np.append(map_cells[0], R_pose_cells[0])
        y = np.append(map_cells[1], R_pose_cells[1])
        contors = np.array([y, x]).T.astype(np.int)

        mask = np.zeros_like(self.log_odds_)
        cv2.drawContours(image=mask, contours=[contors], contourIdx=0, color=np.log(self.p_false_),
                         thickness=cv2.FILLED)

        self.log_odds_ += mask

        MAP['map'] = self.log_odds_

        self.MAP_ = MAP
        return MAP


if __name__ == "__main__":
    slam_inc = SLAM()
    slam_inc._read_data('data/train', 0, 'train')
    slam_inc._init_particles(num_p=100)
    slam_inc._init_map()
    slam_inc._build_first_map()
    MAP = gf.genMap(slam_inc, end_t=1)
    plt.imshow(MAP)
    plt.show()


