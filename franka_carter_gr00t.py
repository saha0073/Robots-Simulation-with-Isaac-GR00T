from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys
import carb
import numpy as np
import torch
import torch.nn as nn
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from safetensors.torch import load_file

# GR00T Policy class based on config.json
class Gr00tPolicy:
    def __init__(self, model_path="E:/Projects/Robotics/gr00t_model/model.safetensors", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.state_dict = load_file(model_path)
        self.model = self._build_model()
        self.model.load_state_dict(self.state_dict, strict=False)  # Non-strict for partial match
        self.model.to(self.device)
        self.model.eval()
        # Linear projection to match Franka's 9 DOF
        self.output_proj = nn.Linear(32, 9).to(self.device)

    def _build_model(self):
        # Simplified GR00T N1 structure from config.json
        class GR00TModel(nn.Module):
            def __init__(self, input_dim=9, hidden_size=1024, num_layers=16, num_heads=32, action_dim=32):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, hidden_size)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.2,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.action_head = nn.Linear(hidden_size, action_dim)

            def forward(self, x):
                x = self.input_proj(x).unsqueeze(1)  # [batch, seq=1, hidden]
                x = self.transformer(x)
                x = self.action_head(x[:, -1, :])  # Last timestep: [batch, action_dim]
                return x

        return GR00TModel()

    def predict(self, joint_states):
        if joint_states.ndim > 1:
            joint_states = joint_states.flatten()
        input_tensor = torch.tensor(joint_states, dtype=torch.float32).to(self.device).unsqueeze(0)  # [1, 9]
        with torch.no_grad():
            action = self.model(input_tensor)  # [1, 32]
            action = self.output_proj(action)  # [1, 9]
        return action.squeeze(0).cpu().numpy()  # [9]

# Scene setup
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()
set_camera_view(eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp")

# Add Franka
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Arm")
arm = Articulation(prim_paths_expr="/World/Arm", name="my_arm")

# Add Carter
asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Carter/nova_carter/nova_carter.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Car")
car = Articulation(prim_paths_expr="/World/Car", name="my_car")

# Set initial poses
arm.set_world_poses(positions=np.array([[0.0, 1.0, 0.0]]) / get_stage_units())
car.set_world_poses(positions=np.array([[0.0, -1.0, 0.0]]) / get_stage_units())

# Load GR00T policy
gr00t_policy = Gr00tPolicy("E:/Projects/Robotics/gr00t_model/model.safetensors")

# Initialize the world
my_world.reset()

# Simulation loop
for i in range(4):
    print("running cycle: ", i)
    if i == 1 or i == 3:
        print("moving")
        car.set_joint_velocities([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    if i == 2:
        print("stopping")
        car.set_joint_velocities([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    for j in range(100):
        arm_joint_states = arm.get_joint_positions()
        if arm_joint_states is not None and (i == 1 or i == 3):
            gr00t_action = gr00t_policy.predict(arm_joint_states)
            arm.set_joint_positions(gr00t_action)
            print(f"GR00T action for arm: {gr00t_action}")
        elif i == 2:
            arm.set_joint_positions([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        my_world.step(render=True)
        if i == 3:
            car_joint_positions = car.get_joint_positions()
            print("car joint positions:", car_joint_positions)

simulation_app.close()