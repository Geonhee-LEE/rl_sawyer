from collections import OrderedDict
import numpy as np
from sawyer_control.core.serializable import Serializable
from sawyer_control.core.eval_util import get_stat_in_paths, \
    create_stats_ordered_dict
from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase

import rospy
from visualization_msgs.msg import Marker
from std_srvs.srv import *


class DDPGEnv(SawyerReachXYZEnv):
    def __init__(self,
                 target_goal=(0, 0, 0),
                 indicator_threshold=.05,
                 reward_type='hand_distance',
                 action_mode='torque',
                 use_safety_box=True,
                 torque_action_scale=1,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())
        # inherit from SawyerReachXYZEnv
        SawyerReachXYZEnv.__init__(self)
        self.torque_action_scale = torque_action_scale
        self.ros_init()
        self.action_mode = action_mode
        if self.action_mode=='torque':
            self.goal_space = self.config.TORQUE_SAFETY_BOX
        else:
            self.goal_space = self.config.POSITION_SAFETY_BOX
        self.indicator_threshold=indicator_threshold
        self.reward_type = reward_type
        self._target_goal = np.array(target_goal)
        self.step_cnt = 0
        

    def ros_init(self):
        # to get marker pos, it is changed when 'done' service triggers.
        rospy.Subscriber("visualization_marker", Marker, self.target_pos_cb)

    def done_client(self):
            #rospy.loginfo("trigger done")
            rospy.wait_for_service('done')
            try:
                done = rospy.ServiceProxy('done', Empty)
                # invoke
                done()
                return 
            except rospy.ServiceException as e:
                print ("Service call failed: ", e)

    def target_pos_cb(self, marker):
        # ros node is already defined at base env 
        self._target_goal=np.array([marker.pose.position.x, 
                                    marker.pose.position.y, 
                                    marker.pose.position.z])
        #print ('_target_goal: ', self._target_goal)

    def step(self, action):
        self.step_cnt += 1
        action = self.convert_input_into_joint_torque(action) # we can only choose few action output
        self._act(action)
        observation = self._get_obs()
        eef_pos = self._get_endeffector_pose()
        self.reward = self.compute_dist_rewards(eef_pos, self._target_goal)
        info = self._get_info()
        self.done = self.check_done()

        if self.step_cnt >= 150: # initialization cnt
            self.step_cnt = 0
            self.reset()
            self.done = True
            

        #print ('action: ', action, '\n reward: ', reward, '\n  done: ', self.done)
        return observation, self.reward, self.done, info
    
    def convert_input_into_joint_torque(self, action):
        torque_action = np.array([action[0], action[1], 0,  action[2], 0, 0, 0]) #'right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5','right_j6'
        #torque_action = np.array([-1, -1, 0, -1, 0, 0, 0]) #'right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5','right_j6'
        return torque_action

    def check_done(self):
        if self.distances < 1.5:
            self.reward = 150.0
            self.done_client()
            return  True      
        else:
            return False

    def reset(self):
        self._reset_robot()
        self._state_goal = self.sample_goal()
        return self._get_obs()

    def _reset_robot(self):
        self.in_reset = True
        self._safe_move_to_neutral()
        self.in_reset = False

    def _safe_move_to_neutral(self):
        #self.send_gripper_cmd("open")
        delay_cnt = 15 #reset delay cnt
        for i in range(delay_cnt):
            cur_pos, cur_vel, _ = self.request_observation()
            torques = self.AnglePDController._compute_pd_forces(cur_pos, cur_vel)
            self._torque_act(torques)
            if self._reset_complete():
                break

    def _get_obs(self):
        angles, velocities, endpoint_pose = self.request_observation()
        obs = np.hstack((
            self._wrap_angles(angles),
            velocities,
            endpoint_pose,
        ))
        return obs

    def _get_endeffector_pose(self):
        _, _, endpoint_pose = self.request_observation()
        return endpoint_pose[:3]

    def _wrap_angles(self, angles):
        return angles % (2*np.pi)

    def compute_dist_rewards(self, eef_pos, goals):
        #print ('goals: ', goals)
        #print ('eef_pos: ', eef_pos)
        #self.distances = np.linalg.norm(eef_pos - goals, axis=0)
        self.distances = np.exp(np.linalg.norm(eef_pos - goals, axis=0))

        if self.reward_type == 'hand_distance':
            r = -self.distances
        elif self.reward_type == 'hand_success':
            r = -(self.distances < self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def _get_info(self):
        hand_distance = np.linalg.norm(self._target_goal - self._get_endeffector_pose())
        return dict(
            hand_distance=hand_distance,
            hand_success=(hand_distance<self.indicator_threshold).astype(float)
        )

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'hand_success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    def _act(self, action):
        if self.action_mode == 'position':
            self._position_act(action * self.position_action_scale)
        else:
            #self._torque_act(action*self.torque_action_scale)
            self._policy_torque_act(action*self.torque_action_scale)
        return

    def _position_act(self, action):
        ee_pos = self._get_endeffector_pose()
        endeffector_pos = ee_pos[:3]
        endeffector_angles = ee_pos[3:]
        target_ee_pos = (endeffector_pos + action)
        target_ee_pos = np.clip(target_ee_pos, self.config.POSITION_SAFETY_BOX_LOWS, self.config.POSITION_SAFETY_BOX_HIGHS)
        target_ee_pos = np.concatenate((target_ee_pos, endeffector_angles))
        angles = self.request_ik_angles(target_ee_pos, self._get_joint_angles())
        print ('position control angles: ', angles)
        self.send_angle_action(angles)

    # move to the goal position using torque neural network
    def _policy_torque_act(self, action):
        self.send_action(action)
        self.rate.sleep()

    # reset torque control
    def _torque_act(self, action):
        if self.use_safety_box:
            if self.in_reset:
                safety_box = self.config.RESET_SAFETY_BOX
            else:
                safety_box = self.config.TORQUE_SAFETY_BOX
            self.get_latest_pose_jacobian_dict()
            pose_jacobian_dict_of_joints_not_in_box = self.get_pose_jacobian_dict_of_joints_not_in_box(safety_box)
            
            if len(pose_jacobian_dict_of_joints_not_in_box) > 0:
                #print('pose_jacobian_dict_of_joints_not_in_box: ', pose_jacobian_dict_of_joints_not_in_box)
                forces_dict = self._get_adjustment_forces_per_joint_dict(pose_jacobian_dict_of_joints_not_in_box, safety_box)
                torques = np.zeros(7)
                for joint in forces_dict:
                    jacobian = pose_jacobian_dict_of_joints_not_in_box[joint][1]
                    force = forces_dict[joint]
                    torques = torques + np.dot(jacobian.T, force).T
                torques[-1] = 0 #we don't need to move the wrist
                action = torques
            

        if self.in_reset: # do resetting
            action = np.clip(action, self.config.RESET_TORQUE_LOW, self.config.RESET_TORQUE_HIGH)
        else:
            action = np.clip(np.asarray(action), self.config.JOINT_TORQUE_LOW, self.config.JOINT_TORQUE_HIGH)
        
        self.send_action(action)
        self.rate.sleep()

    """
    Multitask functions
    """

    @property
    def goal_dim(self):
        return 3

    def convert_obs_to_goals(self, obs):
        return obs[:, -3:]

    def set_to_goal(self, goal):
        self._position_act(goal - self._get_endeffector_pose()[:3])

        # for _ in range(10):
        #     action = goal - self._get_endeffector_pose()[:3]
        #     self._position_act(action * 0.001)    
