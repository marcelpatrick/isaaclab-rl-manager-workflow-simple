# isaaclab-rl-manager-workflow-simple

- This readme provides a more beginner-friendly walkthrough of official tutorials by Nvidia and other sources, such as LycheeAI's, especially [Video2](https://youtu.be/oss4n1NWDKo?si=3gmmHJ2u4bc5DOaT) and [Video4](https://youtu.be/BSQEYj3Wm0Q?si=mMI5WO2w_XQTAVUh).   
- It uses the IsaacLab project **GitHub Repo**: https://github.com/isaac-sim/IsaacLab 
- It goes through the main Python files in this IsaacLab GitHub repo and explains what they do and how they connect.

**Objective**: The goal of this project is to render a cartpole simulation on IsaacSim (using a manager workflow) and use Reinforcement Learning to train the agents in this simulation.

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

**1.1.0: Import Libraries**

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

**1.1.1: Scene Design and Configuration: CartpoleSceneCfg()**

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

**1.2.1: Action Configuration Class: `ActionsCfg`**
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

**1.1.2: Markov Decision Process (MDP)**

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

# 2. RUN TRAINING: `train.py`
`C:\Users\[YOUR USER]\isaaclab\scripts\reinforcement_learning\rl_games\train.py`

To continue the comparison with video game, this is when we push Play and the characters start performing their actions on the level. Or when the crew ***builds the scenario***, allocates the props and the actors ***start the rehearsal***.
This is when we implement Reinforcement Learning and the algorithm starts learning with the results from the actions performed by the agents.

To run the training, we need to: 
- 1. Select the environment we need to train: `_init_.py` > `gym.register(id="[NAME OF THE ENVIRONMENT")`
- 2. Select the algorithm we want to use: `_init_.py` > `kwargs={[TYPE OF ALGORITHM]}`
- 3. Specify the number of envs we want to train: In the VScode terminal: `--num_envs x`
- 4. Run the command (eg for managed based env): `python scripts\reinforcement_learning\skrl\train.py --task Isaac-Velocity-Rough-Anymal-C-v0 num_env 4`


### Gymnasium: 
https://gymnasium.farama.org/index.html 

In the world of Reinforcement Learning (RL), you have two distinct sides:
1. The Environment: The world where the robot lives (Isaac Sim/Isaac Lab). This involves complex physics, friction, lighting, and USD stages.
2. The Agent: The AI brain (neural network) that wants to control the robot. It doesn't know anything about physics or 3D rendering; it only understands numbers (data inputs and action outputs).

**Gymnasium** is a library that sits in the middle so that any Agent can interact with Environments without needing to know how the physics engine works. 
It wraps the code needed to run any environment inside a class and provides standard API functions that allow users to run the main actions needed to perform a simulation.
With this, I can easily switch the agents being tested on the environment, or switch the environments on which I'm testing my agents.
With Gymnasium: All environments expose the same interface and get the same function names (env.reset(), env.step(action))
Plug-and-play: Agents work with any Gymnasium-compliant environment instantly

Without it, every environment (Isaac Lab, MuJoCo, PyBullet) has different APIs: different function names, data formats, observation structures. So Each RL library (Stable Baselines, RSL RL, RL Games) would need custom code for every simulator
Agents can't be reused across environments without rewriting integration code
Without Gymnasium, you'd need custom glue code for every agent-environment pair


- Registration: Once your code is registered within Gymnasium, it can be easily accessed from anywhere by using these templated API calls:

- Register: `gym.register(id, entry_point, **kwargs)` — add an environment name and how to create it so others can instantiate it by that name.
- Make: `gym.make(id, **kwargs)` — create an environment instance from a registered name with one call.
- Reset: `env.reset()` — start or restart an episode and return the initial observation (and info).
- Step: `env.step(action)` — apply an action, advance the sim in time, and return (observation, reward, terminated, truncated, info).
- Close: `env.close()` — release windows/processes/resources used by the environment.
- Spaces: `env.observation_space / env.action_space` — describe the shape, type and bounds of observations/actions so agents format data correctly.
- Render: `render()` shows or returns a visual frame of the environment so you can see what the simulator is doing (for debugging, recording, or human viewing).
- Wrappers: `gym.wrappers.* (e.g., RecordVideo, TimeLimit)` — add recording, time limits, or transforms. Allows users to modify or adapt its interface without changing the original code

<img width="2025" height="484" alt="image" src="https://github.com/user-attachments/assets/7a053c3a-1520-481b-be4c-48bb5f1ccb67" />

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


