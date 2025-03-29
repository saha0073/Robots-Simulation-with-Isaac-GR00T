# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Interactive demo with a single H1 robot in rough terrain, controlled by GR00T N1.

.. code-block:: bash

    # Usage
    C:/Users/subho/issaclab_3.10/Scripts/python.exe E:/Projects/Robotics/h1_groot_locomotion.py

"""

import argparse
import os

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(
    description="Interactive demo with a single H1 robot in rough terrain, controlled by GR00T N1."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import torch.nn as nn
import numpy as np

import carb
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf
from safetensors.torch import load_file

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY

class Gr00tPolicy:
    """GR00T N1 policy adapted for H1 locomotion."""
    def __init__(self, model_path="E:/Projects/Robotics/gr00t_model/model.safetensors", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.state_dict = load_file(model_path)
        
        # Update input dimension to match observation space
        self.input_dim = 256
        self.model = self._build_model()
        self.model.load_state_dict(self.state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Match H1's action space more precisely
        self.output_proj = nn.Linear(32, 19).to(self.device)
        
        # Add action normalization parameters (similar to the original implementation)
        self.action_mean = torch.zeros(19, device=self.device)
        self.action_std = torch.ones(19, device=self.device)
        self.action_scale = 0.1  # Start with very conservative actions
        
        # Fix the default joint angles to match action dimension (19)
        self.default_joint_angles = torch.tensor([
            0.0,  # root
            0.0, 0.0, 1.0,  # root rotation (approximately standing)
            0.0, 0.8, -1.6,  # right hip
            0.0, 0.8, -1.6,  # left hip
            0.0, -0.05,  # right knee
            0.0, -0.05,  # left knee
            0.0, 0.0,  # right ankle
            0.0, 0.0,  # left ankle
            0.0,  # extra joint or control parameter
        ], device=self.device)
        
        # Verify dimensions
        assert self.default_joint_angles.shape[0] == 19, f"Default joint angles should have 19 dimensions, got {self.default_joint_angles.shape[0]}"
        
        # Gradually blend actions
        self.action_filter_coefficient = 0.5
        self.previous_action = None

    def _build_model(self):
        class GR00TModel(nn.Module):
            def __init__(self, input_dim=256):  # Updated input_dim
                super().__init__()
                # Match config parameters
                self.hidden_size = 1536
                self.num_layers = 16
                self.num_heads = 32
                self.head_dim = 48
                self.action_dim = 32
                
                # Input projection to match hidden size
                self.input_proj = nn.Linear(input_dim, self.hidden_size)
                
                # Transformer layers with correct dimensions
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=self.num_heads,
                    dim_feedforward=self.hidden_size * 4,
                    dropout=0.2,
                    batch_first=True,
                    norm_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                
                # Action head to match action dimension
                self.action_head = nn.Linear(self.hidden_size, self.action_dim)

            def forward(self, x):
                # Handle input shape
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # Add sequence dimension
                elif x.dim() == 1:
                    x = x.unsqueeze(0).unsqueeze(1)  # Add batch and sequence dimensions
                
                # Project input to hidden dimension
                x = self.input_proj(x)  # [batch, seq, hidden]
                
                # Apply transformer
                x = self.transformer(x)
                
                # Get action output
                x = self.action_head(x[:, -1, :])  # [batch, action_dim]
                return x

        return GR00TModel(input_dim=self.input_dim)

    def predict(self, obs):
        print(f"Observation min/max: {obs.min():.3f}/{obs.max():.3f}")
        if isinstance(obs, np.ndarray):
            input_tensor = torch.from_numpy(obs).to(dtype=torch.float32, device=self.device)
        else:
            input_tensor = obs.clone().detach().to(dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            # Get raw action from model
            action = self.model(input_tensor)  # [1, 32]
            action = self.output_proj(action)  # [1, 19]
            
            print(f"Raw action shape: {action.shape}")
            print(f"Raw action min/max: {action.min():.3f}/{action.max():.3f}")
            
            # Normalize and scale actions
            action = torch.tanh(action)  # Bound actions to [-1, 1]
            action = action * self.action_scale  # Scale down action magnitude
            
            # Blend with default pose
            action = action + 0.1 * self.default_joint_angles  # Small bias towards default pose
            
            # Action smoothing
            if self.previous_action is not None:
                action = self.action_filter_coefficient * action + \
                        (1 - self.action_filter_coefficient) * self.previous_action
            self.previous_action = action.clone()
            
            print(f"Final action shape: {action.shape}")
            print(f"Final action min/max: {action.min():.3f}/{action.max():.3f}")
            
        return action.squeeze(0).cpu().numpy()

class H1RoughDemo:
    """Interactive demo for a single H1 robot with GR00T control."""
    def __init__(self):
        env_cfg = H1RoughEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1
        env_cfg.curriculum = None
        
        # Use very conservative initial settings
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.3)  # Start very slow
        env_cfg.commands.base_velocity.ranges.heading = (-0.2, 0.2)
        
        # Try to access these config parameters if available
        try:
            env_cfg.control.stiffness = 60.0  # Increase joint stiffness
            env_cfg.control.damping = 6.0     # Increase joint damping
            env_cfg.terrain.max_height = 0.05  # Start with very flat terrain
            env_cfg.terrain.slope_range = (-0.05, 0.05)
        except AttributeError:
            print("Some config parameters were not available")
        
        self.env = ManagerBasedRLEnv(cfg=env_cfg)
        
        # Print environment action space info
        print(f"Environment action space shape: {self.env.action_space.shape}")
        
        self.device = self.env.device
        self.policy = Gr00tPolicy("E:/Projects/Robotics/gr00t_model/model.safetensors")
        
        # Initialize with zero velocity
        self.create_camera()
        self.commands = torch.zeros(1, 4, device=self.device)
        
        # Set up very conservative keyboard controls
        self.set_up_keyboard()

    def create_camera(self):
        stage = omni.usd.get_context().get_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def set_up_keyboard(self):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        
        # Very conservative initial controls
        T = 0.2  # Very slow forward speed
        R = 0.1  # Very slow turning
        
        self._key_to_control = {
            "UP": torch.tensor([T, 0.0, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([-T/2, 0.0, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([T/2, 0.0, 0.0, -R], device=self.device),
            "RIGHT": torch.tensor([T/2, 0.0, 0.0, R], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device),
        }

    def _on_keyboard_event(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._key_to_control:
                self.commands[0] = self._key_to_control[event.input.name]
            elif event.input.name == "C":
                if self.viewport.get_active_camera() == self.camera_path:
                    self.viewport.set_active_camera(self.perspective_path)
                else:
                    self.viewport.set_active_camera(self.camera_path)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.commands[0] = self._key_to_control["ZEROS"]

    def _update_camera(self):
        base_pos = self.env.scene["robot"].data.root_pos_w[0]
        base_quat = self.env.scene["robot"].data.root_quat_w[0]
        camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)
        camera_pos = quat_apply(base_quat, camera_local_transform) + base_pos
        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)

def main():
    demo_h1 = H1RoughDemo()
    obs_dict, _ = demo_h1.env.reset()  # Dictionary output
    obs = obs_dict["policy"]  # Extract policy obs
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation device: {obs.device}")
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # Make sure we're passing the correct observation shape
            if obs.dim() == 2:  # If shape is [1, 256]
                current_obs = obs[0]
            else:  # If shape is [256]
                current_obs = obs
                
            action = demo_h1.policy.predict(current_obs)  # [19]
            print(f"Action shape: {action.shape}")
            
            action_tensor = torch.as_tensor(action, device=demo_h1.device).unsqueeze(0)
            print(f"Action tensor shape: {action_tensor.shape}")
            
            try:
                step_result = demo_h1.env.step(action_tensor)
                print(f"Step result type: {type(step_result)}")
                if isinstance(step_result, tuple):
                    print(f"Step result length: {len(step_result)}")
                    obs_dict = step_result[0]  # Get only the observation dictionary
                else:
                    obs_dict = step_result  # If step returns just the observation dictionary
            except Exception as e:
                print(f"Error during environment step: {e}")
                break
                
            obs = obs_dict["policy"]  # Update obs
            
            # Update commands based on observation shape
            if obs.dim() == 2:
                obs[0, 9:13] = demo_h1.commands[0]  # Inject keyboard commands
            else:
                obs[9:13] = demo_h1.commands[0]
                
        demo_h1._update_camera()

if __name__ == "__main__":
    main()
    simulation_app.close()