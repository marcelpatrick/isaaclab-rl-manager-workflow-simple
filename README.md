# isaaclab-rl-manager-workflow-simple

- This readme provides a more beginner-friendly walkthrough of official tutorials by Nvidia and other sources, such as LycheeAI's, especially [Video2](https://youtu.be/oss4n1NWDKo?si=3gmmHJ2u4bc5DOaT) and [Video4](https://youtu.be/BSQEYj3Wm0Q?si=mMI5WO2w_XQTAVUh).   
- It uses the IsaacLab project **GitHub Repo**: https://github.com/isaac-sim/IsaacLab 
- It goes through the main Python files in this IsaacLab GitHub repo and explains what they do and how they connect.

**Objective**: The goal of this project is to render a cartpole simulation on IsaacSim (using a manager workflow) and use Reinforcement Learning to train the agents in this simulation.

It uses **IsaacLab**
- Isaaclab is a library that automates (with code) the training of robotics simulations on isaacsim.
- Without IsaacLab, you would be able to make 3d world simulations with 3d assets but not use it with RL to train robot policies.
- Isaac Lab is pre-written template and boilerplate code from NVIDIA that handles:
  - Environment templates (cartpole, humanoid configs, task definitions)
  - Manager templates (action/observation/reward processing)
  - RL framework adapters (wrappers for RL-Games, skrl, SB3)
  - Utility code (logging, checkpointing, asset loading)

**Code Map: General**: Here is what the main blocks of code do and how they connect:

<img width="2039" height="3317" alt="image" src="https://github.com/user-attachments/assets/dcc1f51b-75b8-423f-ae0e-e81378102268" />

# 0. Environment Setup

- Open the Anaconda Prompt terminal and activate the conda env you have created with isaacsim and isaaclab installed and all the dependencies and the isaacsim/isaaclab base github project: ``conda activate env_isaacsim``
- If you haven't done it yet you can follow this tutorial: https://github.com/marcelpatrick/IsaacSim-IsaacLab-installation-for-Windows-Easy-Tutorial
-Navigate to the folder containing all the scripts you need to run and type ``code .`` to open VS Code from inside your anaconda env. OR, in this case, after activating the environment, just type: ``code Isaaclab``
  - This will open VS Code with the correct python interpreter from this env and the VS code terminal will also run inside this env. 
- On the folder structure on the left, navigate to the isaaclab project or tutorial you want to run
- click the "run python file" button on VS code to run the script.


# 1. SIMULATION SETUP (CONFIGURATIONS): `cartpole_env_cfg.py`
`C:\Users\[YOUR USER]\isaaclab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\cartpole\cartpole_env_cfg.py`

If we were to think about a simulation as a video game, the first steps we would need to perform to create the game would be to design the player/character, the level where it will play, and the game rules, which define what happens when it performs these actions - eg: when it scores points, when it dies, etc. 

Or to use a movie comparison, we first need to design the scenario and cast the actors before rehearsing and shooting.

The following code is used to:
 1. Configure the environment ("level design" or "set design") and the robot ("character design")
 2. Define the reward function: Markov Decision Process - MDP ("Game Rules")
 
At this point, we will only create the simulation but not learn yet. This will be done in the future steps.
 
Code Map: 
* file: **`cartpole_env_cfg.py`**: Simulation config file
  
  *   **1. Build the Environment:**
    *   `class CartpoleSceneCfg`: Spawns the assets (Ground Plane, Dome Light, robot/Cartpole) 

  *   **2. Reward Function: MDP**
    *   `class ObservationsCfg`: Measures the new robot state after it has performed an action. Sends it back to the AI. Will become the next input the model will use to learn.
    *   `class EventCfg`: Resets robot position after each episode termination (action cycle)
    *   `class RewardsCfg`: Calculates the reward function: (+1) for "staying alive" (-2) for termination: pole tilt, high velocities.
    *   `class TerminationsCfg`: Defines the "Game Over" conditions. It stops the episode if the `time_out` is reached or if the cart moves out of bounds.
    *   `class CartpoleEnvCfg`: Bundles all the below configurations (Scene, Observations, Rewards, etc.) into a single object

## 1.1: Building the Environment

**Import Libraries**

```py
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
```

**Scene Design and Configuration: CartpoleSceneCfg()**

- This is when we ***design the level/scenario*** and ***cast the actors*** that will perform.
- Objects (entities) configuration: Ground, Lights and the Cartpole
- **CARTPOLE_CFG**: the instance of the predefined cartpole configuration object (``from isaaclab_assets.robots.cartpole import CARTPOLE_CFG``) that defines the robot's basic attributes (joints, links, limits, physics).

```py
##
# Scene definition
##


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # cartpole
    # Create a variable "robot" of type ArticulationCfg (annotation), assign to it a configuration copied from the default config template "CARTPOLE_CFG", and save it to this "path" whenever it gets rendered. 
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
```

## 1.2: Reward Function 

- This is when we define the ***Game Rules***

**Action Configuration Class: `ActionsCfg`**
- Converts the RL action output (a float) into a physical force/effort applied to the chosen joint (in Newtons).
- eg: So the policy outputs, for example, 0.3, and the action definition turns that into: 0.3 × scale (100) = 30 units of joint effort applied to the slider_to_cart joint

```py

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # action definition: “The agent controls the cart by applying effort/force to the slider_to_cart joint of the robot,” scaled by 100 so the RL policy’s output maps to meaningful physical forces
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
```

### Markov Decision Process (MDP)
MDP is a mathematical model that, given an agent's current state and its action, defines its what happens to it, updates its new state and calculates a reward for the result of its action. 
The RL algorithm (skrl, rl_games etc) will then take the result of the reward function to learn and suggest the next best action for the agent that optimizes the reward function. 

***Observations Configuration Class: `ObservationsCfg`***
- Inputs into the deep network (X)
- It defines what information the robot’s brain (the RL policy) gets in each step.
- “Collect the robot’s joint positions and velocities, package them into one vector, don’t add noise, and feed that to the policy every step.” So the RL agent learns using only those two signals about the cart-pole’s state: joint position ``joint_pos_rel``, joint velocity ``joint_vel_rel``
  
```py
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # observes the relative joint position of all joints
        # "func" method searches through joints IDs. 
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
```

***Event Configuration Class:*** `EventCfg`
- It defines how the robot resets at the start of each episode.
- Each EventTerm randomizes the cart and pole joint positions and velocities within given ranges, ensuring varied starting states so the RL agent learns a more robust policy.

```py
@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )
```

***Reward Configuration Class:***`RewardsCfg`
- It defines how the agent is rewarded or penalized.
- The code gives a positive reward for staying alive, penalizes failure, penalizes the pole being off upright, and adds small penalties for cart and pole motion.
- Together, these incentives teach the agent to balance the pole steadily.

```py
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # is_alive and is_terminated are predefined helper functions inside the isaaclab_tasks.manager_based.classic.cartpole.mdp module.
    # They are not generic Python or Isaac Lab functions; they are task-specific MDP utilities provided by the cartpole MDP implementation to detect success or failure conditions.

    # POSITIVE REINFORCEMENT: REWARD: weight=1.0
    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # NEGATIVE REINFORCEMENT: REWARD: weight=-2.0
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # NEGATIVE REINFORCEMENT: REWARD: weight=-1.0
    # (3) Primary task: keep pole upright
    # Punishes whenever the pole has position deviations away from the upright position
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )

    # NEGATIVE REINFORCEMENT: REWARD: weight=-0.01
    # (4) Shaping tasks: lower cart velocity
    # Punishes if the robot speeds too much
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )

    # NEGATIVE REINFORCEMENT: REWARD: weight=-0.005
    # (5) Shaping tasks: lower pole angular velocity
    # Punishes whenever the pole acquires angular velocities which are too high
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )
```

***Termination Class Configuration:*** `TerminationsCfg`
- It defines when an episode should end.
- One rule ends the episode after a time limit; the other ends it if the cart moves outside the allowed range.
- These termination conditions tell the RL system when to reset and start a new episode.
- **Episode**: is a sequence of interactions between the agent and the environment. When the agent finishes its "mission" the key sequence of actions it was predefined to perform in order to learn. After each Episode, the accumulated rewards are calculated and the result is used to train the algorithm -> back-propagation. 

```py
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )
```

***Carpole Environment Configuration Class:*** `CartpoleEnvCfg`

- It bundles all config classes above and returns one single config object that tells Isaac Lab how to build and run the full RL environment.
- It also sets global parameters like episode length, step rate, rendering interval, and viewer position.

```py
##
# Environment configuration
##


@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0, clone_in_fabric=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
```

**MDP Cycle:**
<img width="1753" height="589" alt="image" src="https://github.com/user-attachments/assets/434ddbb8-6f89-4dd1-b176-87651410e2fa" />

# 2. REGISTER THE ENVIRONMENT ON GYMNASIUM
`C:\Users\[YOUR USER]\isaaclab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\cartpole\__init__.py`
`https://gymnasium.farama.org/index.html `

<img width="2025" height="484" alt="image" src="https://github.com/user-attachments/assets/7a053c3a-1520-481b-be4c-48bb5f1ccb67" />

In the world of Reinforcement Learning (RL), you have two distinct sides:
1. The Environment: The world where the robot lives (Isaac Sim/Isaac Lab). This involves complex physics, friction, lighting, and USD stages.
2. The Agent: The AI brain (neural network) that wants to control the robot. It doesn't know anything about physics or 3D rendering; it only understands numbers (data inputs and action outputs).
**Gymnasium** is a library that sits in the middle so that any Agent can interact with Environments without needing to know how the physics engine works. 
It wraps the code needed to run any environment inside a class and provides standard API functions that allow users to run the main actions needed to perform a simulation.
With this, I can easily switch the agents being tested on the environment, or switch the environments on which I'm testing my agents.
- With Gymnasium: All environments expose the same interface and get the same function names (env.reset(), env.step(action))
Plug-and-play: Agents work with any Gymnasium-compliant environment instantly
- Without it, every environment (Isaac Lab, MuJoCo, PyBullet) has different APIs: different function names, data formats, observation structures. So Each RL library (Stable Baselines, RSL RL, RL Games) would need custom code for every simulator
Agents can't be reused across environments without rewriting integration code
Without Gymnasium, you'd need custom glue code for every agent-environment pair

Some of the Gymnasium custom functions:
- Register: `gym.register(id, entry_point, **kwargs)` — add an environment name and how to create it so others can instantiate it by that name.
- Make: `gym.make(id, **kwargs)` — create an environment instance from a registered name with one call.
- Reset: `env.reset()` — start or restart an episode and return the initial observation (and info).
- Step: `env.step(action)` — apply an action, advance the sim in time, and return (observation, reward, terminated, truncated, info).
- Close: `env.close()` — release windows/processes/resources used by the environment.
- Spaces: `env.observation_space / env.action_space` — describe the shape, type and bounds of observations/actions so agents format data correctly.
- Wrappers: `gym.wrappers.* (e.g., RecordVideo, TimeLimit)` — add recording, time limits, or transforms. Allows users to modify or adapt its interface without changing the original code

- Once your code is registered within Gymnasium, it can be easily accessed from anywhere by using these templated API calls.

- `_init_.py`: converts the Python folders in this project into a package.
  - This makes it easier for users to import functions implemented by the code in this folder and provides callable public APIs.
  - It's used to import the `CartpoleEnvCfg` class, which is used to generate the env config object
  - It also registers this project into Gymnasium: It tells the Gymnasium interface which env config class to import: `entry_point=f"{__name__}.cartpole_env:CartpoleEnv"`

- This particular `_init_.py` file is registering these 5 envs by their names: "Isaac-Cartpole-v0", "Isaac-Cartpole-RGB-v0", "Isaac-Cartpole-Depth-v0", "Isaac-Cartpole-RGB-ResNet18-v0", "Isaac-Cartpole-RGB-TheiaTiny-v0",
 
```py
"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Cartpole-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerWithSymmetryCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleRGBCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleDepthCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-RGB-ResNet18-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleResNet18CameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_feature_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-RGB-TheiaTiny-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleTheiaTinyCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_feature_ppo_cfg.yaml",
    },
)

```

# 3. TRAIN: `train.py`
In this example, we'll be using the training script with the RL-Games library: `C:\Users\[YOUR USER]\isaaclab\scripts\reinforcement_learning\rl_games\train.py`

To continue the comparison with video game, this is when we push Play and the characters start performing their actions on the level. Or when the crew ***builds the scenario***, allocates the props and the actors ***start the rehearsal***.
This is when we implement Reinforcement Learning and the algorithm starts learning with the results from the actions performed by the agents.

To run the training, we need to: 
- 1. Select the environment we need to train: `_init_.py` > `gym.register(id="[NAME OF THE ENVIRONMENT")`
- 2. Select the algorithm we want to use: `_init_.py` > `kwargs={[TYPE OF ALGORITHM]}`
- 3. Specify the number of envs we want to train: In the VScode terminal: `--num_envs x`
- 4. Run the command (eg for managed based env): `python scripts\reinforcement_learning\skrl\train.py --task Isaac-Velocity-Rough-Anymal-C-v0 num_env 4`

## 3.1: The training script

`train.py` is a launcher script that boots Isaac Sim, loads an Isaac Lab task via Gymnasium, wraps it for the RL-Games library, and runs training. It orchestrates the entire training pipeline for robot learning tasks.
Each train.py under the reinforcement_learning folder in this project wires Isaac Lab to a different RL library (skrl, rl-games ...), each library using different RL algorithms (PPO, IPPO, MAPPO, AMP etc)

1. The script first builds the simulated world, then env (cartpole, humanoid, etc.) using Gymnasium and Isaac Lab.
2. Then RL-Games decides which actions the agent should perform next (like a "coach"), based on observations of the rewards. It  updates the neural network policy (the "game tactics") to do better.
3. The environment (Isaac Lab/Isaac Sim) applies those actions to the simulated robot, runs physics, and returns the next state and reward.

**PARSE COMMAND-LINE ARGUMENTS**
```py
"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
```

**LAUNCH ISAACSIM**
Uses AppLauncher to boot up the NVIDIA Isaac Sim physics simulator.

```py
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

**IMPORTS**

```py
"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
from datetime import datetime

import omni
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rl_games import MultiObserver, PbtAlgoObserver, RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
```

### MAIN TRAINING FUNCTION

This is the core function that sets up and runs the training process.
It takes configuration settings (loaded from YAML files via the decorator above) and command-line arguments to:
1. Configure the simulation environment (how many robots to simulate, which GPU to use)
2. Set up the AI agent that will learn to control the robot
3. Run the training loop where the agent learns through trial and error

**CONFIGURE THE ENVIRONMENT**
- Loads the environment configuration using Hydra, then overrides with CLI arguments (num_envs, device, seed, max_iterations, etc.).

```py
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""

    # =========================================================================
    # SECTION 4: CONFIGURE THE ENVIRONMENT
    # Loads the environment configuration using Hydra, then overrides with
    # CLI arguments (num_envs, device, seed, max_iterations, etc.).
    # =========================================================================

    # ====================These functions override default env configs with command line args===========
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # update agent device to match simulation device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]
```

**SET UP LOGGING**
- This section creates a folder structure to save all training data and results.
- During training, the AI agent's progress (performance metrics, learned behaviors, configuration settings, and optional video recordings) are saved to these folders.
- This makes it easy to track experiments, compare different training runs, and 
    resume training later if needed.

```py
 # specify directory for logging experiments
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.join("logs", "rl_games", config_name)
    if "pbt" in agent_cfg and agent_cfg["pbt"]["directory"] != ".":
        log_root_path = os.path.join(agent_cfg["pbt"]["directory"], log_root_path)
    else:
        log_root_path = os.path.abspath(log_root_path)

    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
    experiment_name = log_dir if args_cli.wandb_name is None else args_cli.wandb_name

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    print(f"Exact experiment name requested from command line: {os.path.join(log_root_path, log_dir)}")

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = os.path.join(log_root_path, log_dir)
```

**CREATE THE TRAINING ENVIRONMENT**: gym.make()
- Uses `gym.make()` to instantiate the environment, then wraps it for rl-games compatibility and optional video recording.

```py
# Create the Gymnasium environment where the AI agent will learn.
    # The "task" defines what scenario to simulate (e.g., cartpole, robot arm).
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
```

**RUN TRAINING**: runner.run()
- Uses rl-games library Runner class (`runner.run()`) to execute the RL training loop.

```py
 # create runner from rl-games

    if "pbt" in agent_cfg and agent_cfg["pbt"]["enabled"]:
        observers = MultiObserver([IsaacAlgoObserver(), PbtAlgoObserver(agent_cfg, args_cli)])
        runner = Runner(observers)
    else:
        runner = Runner(IsaacAlgoObserver())

    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent

    global_rank = int(os.getenv("RANK", "0"))
    if args_cli.track and global_rank == 0:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb

        wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=experiment_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        if not wandb.run.resumed:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
            wandb.config.update({"agent_cfg": agent_cfg})

    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})
```

**CLEAN UP**: `env.close()`
- Closes the environment and simulation app when done.

```py
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
```

## 3.2: Run Training

Open a Terminal inside VS Code and run the command: `python scripts\reinforcement_learning\rl_games\train.py --task Isaac-Cartpole-v0`
    
- You can also specify the number of envs you want the simulation to run with `--num_envs X`
`python scripts\reinforcement_learning\rl_games\train.py --task Isaac-Cartpole-v0 --num_envs 512`
    
- You can also run different envs that are supported by this project, eg:
  - For Direct mode: `python scripts\reinforcement_learning\skrl\train.py --task Isaac-Humanoid-Direct-v0 --num_envs 4`
  - For Manager mode: `python scripts\reinforcement_learning\skrl\train.py --task Isaac-Velocity-Rough-Anymal-C-v0 --num_envs 4`

- You can select any other environment from the Nvidia list of available environments `https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html` and run them (manager/direct) with: `python scripts\reinforcement_learning\skrl\train.py --task [ENVIRONMENT NAME] --num_envs (x)`
  - Some might not work if they are not implemented/registered in this project `isaaclab`
 

Request and Response flow:
```
train.py                        GYMNASIUM                     __init__.py                      cartpole_env_cfg.py
════════                        ═════════                     ═══════════                      ═══════════════════
    │                               │                              │                                   │
    │  1. import isaaclab_tasks     │                              │                                   │
    │  ────────────────────────────────────────────────────►       │                                   │
    │                               │                        executes gym.register()                   │
    │                               │                        at import time                            │
    │                               │                              │                                   │
    │                               │   2. registers task          │                                   │
    │                               │   ◄──────────────────────────┘                                   │
    │                               │   stores in global registry:                                     │
    │                               │   {"Isaac-Cartpole-v0": ...}                                     │
    │                               │                                                                  │
    │  3. gym.make("Isaac-Cartpole-v0", cfg=env_cfg)                                                   │
    │  ─────────────────────────►   │                                                                  │
    │                               │   4. looks up "Isaac-Cartpole-v0"                                │
    │                               │      in registry                                                 │
    │                               │                              │                                   │
    │                               │   5. parses entry_point:     │                                   │
    │                               │      "isaaclab.envs:ManagerBasedRLEnv"                           │
    │                               │                              │                                   │
    │                               │   6. parses env_cfg_entry_point                                  │
    │                               │   ─────────────────────────────────────────────────────────────► │
    │                               │                              │                           imports │
    │                               │                              │                     CartpoleEnvCfg│
    │                               │   ◄───────────────────────────────────────────────────────────── │
    │                               │                              │                    returns config │
    │                               │                                                                  │
    │                               │   7. instantiates:                                               │
    │                               │      ManagerBasedRLEnv(cfg=CartpoleEnvCfg)                       │
    │                               │                                                                  │
    │  ◄────────────────────────────┘                                                                  │
    │  8. returns env instance                                                                         │
    │                                                                                                  │
    │  9. RlGamesVecEnvWrapper(env)                                                                    │
    │     wraps env for RL-Games                                                                       │
    │                                                                                                  │
    │  10. runner.run() → training loop                                                                │
    │      calls env.step() repeatedly                                                                 │
    ▼                                                                                                  │
 TRAINING                                                                                              │
```


