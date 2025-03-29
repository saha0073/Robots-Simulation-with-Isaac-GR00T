"""
This script demonstrates different dexterous hands controlled by GR00T.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/hands_gr00t.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import load_file

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different dexterous hands.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

##
# Pre-defined configs
##
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG

class Gr00tPolicy:
    """GR00T N1 policy adapted for dexterous hands."""
    def __init__(self, model_path="E:/Projects/Robotics/gr00t_model/model.safetensors", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.state_dict = load_file(model_path)
        print(f"Loaded state dict keys: {self.state_dict.keys()}")
        
        # Update input dimension to match observation space
        self.input_dim = 256
        self.model = self._build_model()
        self.model.load_state_dict(self.state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Output projections for different hand types
        self.allegro_proj = nn.Linear(32, 16).to(self.device)  # Allegro hand DOF
        self.shadow_proj = nn.Linear(32, 24).to(self.device)   # Shadow hand DOF
        
        # Action smoothing
        self.action_scale = 0.1
        self.action_filter_coefficient = 0.5
        self.previous_actions = {}

    def _build_model(self):
        class GR00TModel(nn.Module):
            def __init__(self, input_dim=256):
                super().__init__()
                self.hidden_size = 1536
                self.num_layers = 16
                self.num_heads = 32
                self.head_dim = 48
                self.action_dim = 32
                
                self.input_proj = nn.Linear(input_dim, self.hidden_size)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=self.num_heads,
                    dim_feedforward=self.hidden_size * 4,
                    dropout=0.2,
                    batch_first=True,
                    norm_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                self.action_head = nn.Linear(self.hidden_size, self.action_dim)

            def forward(self, x):
                if x.dim() == 1:
                    x = x.unsqueeze(0).unsqueeze(1)
                elif x.dim() == 2:
                    x = x.unsqueeze(1)
                
                x = self.input_proj(x)
                x = self.transformer(x)
                x = self.action_head(x[:, -1, :])
                return x

        return GR00TModel(input_dim=self.input_dim)

    def predict(self, obs, hand_type="allegro"):
        if isinstance(obs, np.ndarray):
            input_tensor = torch.from_numpy(obs).to(dtype=torch.float32, device=self.device)
        else:
            input_tensor = obs.clone().detach().to(dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            action = self.model(input_tensor)
            
            # Project to appropriate hand DOF
            if hand_type == "allegro":
                action = self.allegro_proj(action)
            else:  # shadow
                action = self.shadow_proj(action)
            
            # Normalize and scale actions
            action = torch.tanh(action) * self.action_scale
            
            # Action smoothing
            if hand_type not in self.previous_actions:
                self.previous_actions[hand_type] = action.clone()
            else:
                action = self.action_filter_coefficient * action + \
                        (1 - self.action_filter_coefficient) * self.previous_actions[hand_type]
                self.previous_actions[hand_type] = action.clone()
            
        return action.squeeze(0).cpu().numpy()

def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the scene."""
    env_origins = torch.zeros(num_origins, 3)
    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    return env_origins.tolist()

def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origins = define_origins(num_origins=2, spacing=0.5)

    # Origin 1 with Allegro Hand
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    allegro = Articulation(ALLEGRO_HAND_CFG.replace(prim_path="/World/Origin1/Robot"))

    # Origin 2 with Shadow Hand
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    shadow_hand = Articulation(SHADOW_HAND_CFG.replace(prim_path="/World/Origin2/Robot"))

    scene_entities = {
        "allegro": allegro,
        "shadow_hand": shadow_hand,
    }
    return scene_entities, origins

def create_observation(robot, device):
    # Example observation creation
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    # Add other relevant state information
    obs = torch.cat([joint_pos, joint_vel], dim=-1)
    # Pad or process to match 256-dim input
    padded_obs = torch.zeros(256, device=device)
    padded_obs[:obs.shape[-1]] = obs
    return padded_obs

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], 
                 origins: torch.Tensor, policy: Gr00tPolicy):
    """Runs the simulation loop with GR00T policy."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Print model info
    print("[INFO]: Using GR00T model for hand control")
    print(f"[INFO]: Model path: {policy.state_dict.keys()}")

    while simulation_app.is_running():
        if count % 1000 == 0:
            sim_time = 0.0
            count = 0
            for index, (name, robot) in enumerate(entities.items()):
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                robot.write_joint_state_to_sim(
                    robot.data.default_joint_pos.clone(), 
                    robot.data.default_joint_vel.clone()
                )
                robot.reset()
            print("[INFO]: Resetting robots state...")

        # Get observations and apply GR00T actions
        for name, robot in entities.items():
            # Create rich observation for GR00T
            joint_pos = robot.data.joint_pos
            joint_vel = robot.data.joint_vel
            # Add more state information that GR00T might need
            obs = torch.cat([
                joint_pos,  # Current joint positions
                joint_vel,  # Current joint velocities
                robot.data.soft_joint_pos_limits[..., 0],  # Joint limits min
                robot.data.soft_joint_pos_limits[..., 1],  # Joint limits max
            ], dim=-1)
            
            # Pad to 256 dimensions for GR00T
            padded_obs = torch.zeros(256, device=policy.device)
            padded_obs[:obs.shape[-1]] = obs
            
            # Get GR00T action
            try:
                action = policy.predict(padded_obs, hand_type=name)
                print(f"[DEBUG] GR00T action for {name}: min={action.min():.3f}, max={action.max():.3f}")
            except Exception as e:
                print(f"[ERROR] GR00T prediction failed: {e}")
                # Fallback to default behavior
                action = robot.data.soft_joint_pos_limits[..., count % 2].cpu().numpy()
            
            # Apply action
            robot.set_joint_position_target(torch.tensor(action, device=sim.device))
            robot.write_data_to_sim()

        sim.step()
        sim_time += sim_dt
        count += 1
        
        for robot in entities.values():
            robot.update(sim_dt)

def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[0.0, -0.5, 1.5], target=[0.0, -0.2, 0.5])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    
    # Initialize GR00T policy
    policy = Gr00tPolicy(model_path="E:/Projects/Robotics/gr00t_model/model.safetensors", 
                        device=sim.device)
    
    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins, policy)

if __name__ == "__main__":
    main()
    simulation_app.close()
