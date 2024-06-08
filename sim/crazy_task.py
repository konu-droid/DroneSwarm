from gymnasium import spaces
import numpy as np
import torch
import math
import cv2

from typing import Optional

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)

class CarzyFlyTask(BaseTask):
    def __init__(self, name: str, offset: np.ndarray | None = None) -> None:

        # task-specific parameters
        self._fly_position = [0.0, 0.0, 0.2] #X, Y, Z in world coordinates
        self._reset_dist = 5.0
        self._max_push_effort = 400.0

        # values used for defining RL buffers
        self._num_observations = 13
        self._num_actions = 4
        self._device = "cpu"
        self.num_envs = 1
        self.dt = 1.0/10.0 #self._task_cfg["sim"]["dt"]

        # parameters for the crazyflie
        self.arm_length = 0.05

        # parameters for the controller
        self.motor_damp_time_up = 0.15
        self.motor_damp_time_down = 0.15

        # I use the multiplier 4, since 4*T ~ time for a step response to finish, where
        # T is a time constant of the first-order filter
        self.motor_tau_up = 4 * self.dt / (self.motor_damp_time_up + EPS)
        self.motor_tau_down = 4 * self.dt / (self.motor_damp_time_down + EPS)

        # thrust max
        self.mass = 0.028
        self.thrust_to_weight = 1.9

        self.motor_assymetry = np.array([1.0, 1.0, 1.0, 1.0])
        # re-normalizing to sum-up to 4
        self.motor_assymetry = self.motor_assymetry * 4.0 / np.sum(self.motor_assymetry)

        self.grav_z = 9.8

        # a few class buffers to store RL-related states
        self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(
            np.ones(self._num_actions, dtype=np.float32) * -1.0,
            np.ones(self._num_actions, dtype=np.float32) * 1.0,
        )
        self.observation_space = spaces.Box(
            np.ones(self._num_observations, dtype=np.float32) * -np.Inf,
            np.ones(self._num_observations, dtype=np.float32) * np.Inf,
        )

        super().__init__(name, offset)

    def set_up_scene(self, scene) -> None:
        # print(" Innnnnnnnnnnnnnnnnnnnnnnn set_up_scene")
        # retrieve file path for the Carzy fly USD file
        assets_root_path = "/home/konu/Documents/mini_drone/DroneSwarm/sim/usd/"
        usd_path = assets_root_path + "cf2x.usd"

        add_reference_to_stage(usd_path, "/World")
        # create an ArticulationView wrapper for our Carzy fly - this can be extended towards accessing multiple Carzy flies
        self._flies = CrazyflieView(
            prim_paths_expr="/World/cf2x*", name="CrazyflieView"
        )
        # add Carzy fly ArticulationView and ground plane to the Scene
        scene.add(self._flies)
        scene.add_default_ground_plane()
        for i in range(4):
            scene.add(self._flies.physics_rotors[i])

        # set default camera viewport position and target
        self.set_initial_camera_params()

    def set_initial_camera_params(
        self, camera_position=[2, 2, 1], camera_target=[0, 0, 0]
    ):
        set_camera_view(
            eye=camera_position,
            target=camera_target,
            camera_prim_path="/OmniverseKit_Persp",
        )

    def post_reset(self):
        # print(" Innnnnnnnnnnnnnnnnnnnnnnn post_reset")
        thrust_max = self.grav_z * self.mass * self.thrust_to_weight * self.motor_assymetry / 4.0
        self.thrusts = torch.zeros((self.num_envs, 4, 3), dtype=torch.float32, device=self._device)
        self.thrust_cmds_damp = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self._device)
        self.thrust_rot_damp = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self._device)
        self.thrust_max = torch.tensor(thrust_max, device=self._device, dtype=torch.float32)

        self.motor_linearity = 1.0
        self.prop_max_rot = 433.3

        self._fly_m1_dof_idx = self._flies.get_dof_index("m1_joint")
        self._fly_m2_dof_idx = self._flies.get_dof_index("m2_joint")
        self._fly_m3_dof_idx = self._flies.get_dof_index("m3_joint")
        self._fly_m4_dof_idx = self._flies.get_dof_index("m4_joint")

        # print(f"_carter_left_dof_idx: {self._carter_left_dof_idx}, _carter_right_dof_idx: {self._carter_right_dof_idx}")
        # randomize all envs
        self.all_indices = torch.arange(
            self._flies.count, dtype=torch.int64, device=self._device
        )

        self.reset(self.all_indices)


    def reset(self, env_ids=None):
        # print(" Innnnnnnnnnnnnnnnnnnnnnnn reset")
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        num_resets = len(env_ids)

        # randomize DOF velocities
        self.dof_vel[:, self._fly_m1_dof_idx] = 0.1  # 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        self.dof_vel[:, self._fly_m2_dof_idx] = 0.1  # 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        self.dof_vel[:, self._fly_m3_dof_idx] = 0.1  # 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        self.dof_vel[:, self._fly_m4_dof_idx] = 0.1  # 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        fly_positions = torch.zeros((num_resets, 3), device=self._device)
        fly_positions[:, 0] = torch.arange(num_resets)
        fly_positions[:, 2] = 0.1
        fly_orientations = torch.tile(torch.tensor([1.0, 0.0, 0.0, 0.0]), (num_resets, 1))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._flies.set_joint_velocities(self.dof_vel, indices=indices)
        self._flies.set_world_poses(fly_positions, fly_orientations, indices=indices)

        # bookkeeping
        self.resets[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        # print(" Innnnnnnnnnnnnnnnnnnnnnnn pre_physics_step")

        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.clone().to(self._device)
        self.actions = actions

        # clamp to [-1.0, 1.0]
        thrust_cmds = torch.clamp(actions, min=-1.0, max=1.0)
        # scale to [0.0, 1.0]
        thrust_cmds = (thrust_cmds + 1.0) / 2.0
        # filtering the thruster and adding noise
        motor_tau = self.motor_tau_up * torch.ones((self.num_envs, 4), dtype=torch.float32, device=self._device)
        motor_tau[thrust_cmds < self.thrust_cmds_damp] = self.motor_tau_down
        motor_tau[motor_tau > 1.0] = 1.0

        # Since NN commands thrusts we need to convert to rot vel and back
        thrust_rot = thrust_cmds**0.5
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
        self.thrust_cmds_damp = self.thrust_rot_damp**2

        ## Adding noise
        thrust_noise = 0.01 * torch.randn(4, dtype=torch.float32, device=self._device)
        thrust_noise = thrust_cmds * thrust_noise
        self.thrust_cmds_damp = torch.clamp(self.thrust_cmds_damp + thrust_noise, min=0.0, max=1.0)

        thrusts = self.thrust_max * self.thrust_cmds_damp

        # thrusts given rotation
        root_quats = self.root_rot
        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)
        rot_matrix = torch.cat((rot_x, rot_y, rot_z), 1).reshape(-1, 3, 3)

        force_x = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self._device)
        force_y = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self._device)
        force_xy = torch.cat((force_x, force_y), 1).reshape(-1, 4, 2)
        thrusts = thrusts.reshape(-1, 4, 1)
        thrusts = torch.cat((force_xy, thrusts), 2)

        thrusts_0 = thrusts[:, 0]
        thrusts_0 = thrusts_0[:, :, None]

        thrusts_1 = thrusts[:, 1]
        thrusts_1 = thrusts_1[:, :, None]

        thrusts_2 = thrusts[:, 2]
        thrusts_2 = thrusts_2[:, :, None]

        thrusts_3 = thrusts[:, 3]
        thrusts_3 = thrusts_3[:, :, None]

        mod_thrusts_0 = torch.matmul(rot_matrix, thrusts_0)
        mod_thrusts_1 = torch.matmul(rot_matrix, thrusts_1)
        mod_thrusts_2 = torch.matmul(rot_matrix, thrusts_2)
        mod_thrusts_3 = torch.matmul(rot_matrix, thrusts_3)

        self.thrusts[:, 0] = torch.squeeze(mod_thrusts_0)
        self.thrusts[:, 1] = torch.squeeze(mod_thrusts_1)
        self.thrusts[:, 2] = torch.squeeze(mod_thrusts_2)
        self.thrusts[:, 3] = torch.squeeze(mod_thrusts_3)

        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0

        # spin spinning rotors
        prop_rot = self.thrust_cmds_damp * self.prop_max_rot

        dof_vel = torch.zeros(
            (self._flies.count, self._flies.num_dof),
            dtype=torch.float32,
            device=self._device,
        )
        dof_vel[:, self._fly_m1_dof_idx] = prop_rot[:, 0]
        dof_vel[:, self._fly_m2_dof_idx] = -1.0 * prop_rot[:, 1]
        dof_vel[:, self._fly_m3_dof_idx] = prop_rot[:, 2]
        dof_vel[:, self._fly_m4_dof_idx] = -1.0 * prop_rot[:, 3]

        self._flies.set_joint_velocities(dof_vel)

        # apply actions
        for i in range(4):
            self._flies.physics_rotors[i].apply_forces(self.thrusts[:, i], indices=self.all_indices)

    def get_observations(self):
        # # print(" Innnnnnnnnnnnnnnnnnnnnnnn get_observations")
        self.root_pos, self.root_rot = self._flies.get_world_poses(clone=False)
        self.root_velocities = self._flies.get_velocities(clone=False)

        self.obs[..., 0:3] = self.root_pos
        self.obs[..., 3:7] = self.root_rot
        self.obs[..., 7:] = self.root_velocities

        return self.obs

    def calculate_metrics(self) -> None:
        # # print(" Innnnnnnnnnnnnnnnnnnnnnnn calculate_metrics")
        # cart_pos = self.obs[:, 0]
        # cart_vel = self.obs[:, 1]
        # pole_angle = self.obs[:, 2]
        # pole_vel = self.obs[:, 3]

        # # compute reward based on angle of pole and cart velocity
        # reward = (
        #     1.0
        #     - pole_angle * pole_angle
        #     - 0.01 * torch.abs(cart_vel)
        #     - 0.005 * torch.abs(pole_vel)
        # )
        # # apply a penalty if cart is too far from center
        # reward = torch.where(
        #     torch.abs(cart_pos) > self._reset_dist,
        #     torch.ones_like(reward) * -2.0,
        #     reward,
        # )
        # # apply a penalty if pole is too far from upright
        # reward = torch.where(
        #     torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward
        # )

        # return reward.item()
        return True

    def is_done(self) -> None:
        # print(" Innnnnnnnnnnnnnnnnnnnnnnn is_done")
        cart_pos = self.obs[:, 0]

        # reset the robot if cart has reached reset_dist or pole is too far from upright
        resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        self.resets = resets

        # return resets.item()
        return True

class CrazyflieView(ArticulationView):
    def __init__(self, prim_paths_expr: str, name: Optional[str] = "CrazyflieView") -> None:
        """[summary]"""

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
        )

        self.physics_rotors = [
            RigidPrimView(prim_paths_expr=prim_paths_expr+f"/m{i}_prop", name=f"m{i}_prop_view")
            for i in range(1, 5)
        ]