# isaaclab-rl-manager-workflow-simple

- This readme provides a more structured and beginner-friendly walkthrough of official tutorials by Nvidia and other sources. 
- It follows the RL tutorials by LycheeAI. Especially [Video2](https://youtu.be/oss4n1NWDKo?si=3gmmHJ2u4bc5DOaT) and [Video4](https://youtu.be/BSQEYj3Wm0Q?si=mMI5WO2w_XQTAVUh).   
- It uses files from the IsaacLab **GitHub Repo**: https://github.com/isaac-sim/IsaacLab 
- It goes through the main Python files in this IsaacLab GitHub repo and explains what they do.

- **Objective**: The goal of this project is to render a cartpole simulation on IsaacSim (using a manager workflow) and use Reinforcement Learning to train the agents in this simulation.

**Code Map**

<img width="2039" height="3317" alt="image" src="https://github.com/user-attachments/assets/dcc1f51b-75b8-423f-ae0e-e81378102268" />

# 0.Setup

- Open the Anaconda Prompt terminal and activate the conda env you have created with isaacsim and isaaclab installed and all the dependencies and the isaacsim/isaaclab base github project: ``conda activate env_isaacsim``
- If you haven't done it yet you can follow this tutorial: https://github.com/marcelpatrick/IsaacSim-IsaacLab-installation-for-Windows-Easy-Tutorial
-Navigate to the folder containing all the scripts you need to run and type ``code .`` to open VS Code from inside your anaconda env. OR, in this case, after activating the environment, just type: ``code Isaaclab``
  - This will open VS Code with the correct python interpreter from this env and the VS code terminal will also run inside this env. 
- On the folder structure on the left, navigate to the isaaclab project or tutorial you want to run
- click the "run python file" button on VS code to run the script.


# 1. SIMULATION SETUP

The following code does:
 1. Builds the environment ("level design") and the robot ("character design")
 2. Defines the reward function: Markov Decision Process - MDP ("Game Rules")
 3. Define actions robots will perform when the simulation runs, first manually ("player actions")

 
At this point, we will only create the simulation but not learn yet. This will be done in the future steps.
We will also manually run the simulation and MDP for testing, but not yet plug it into the Reinforcement Learning framework.
This simulation will be later registered in Gymnasium which will plug its observations and rewards to an RL algorithm which will enable learning.
 





### Gymnasium: 

<img width="2025" height="484" alt="image" src="https://github.com/user-attachments/assets/7a053c3a-1520-481b-be4c-48bb5f1ccb67" />

```
╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║               ISAAC LAB TRAINING FLOW - CALL DIAGRAM WITH GYMNASIUM (Time ↓)                                                    ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

   train.py    |    GYMNASIUM              │          __init__.py           │        cartpole_env_cfg.py
   ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
      │                 │                                  │                                │
      │                 │  ┌─────── Global Registry ──────┐│                                │
      │                 │  │ Task ID → entry_point        ││                                │
      │                 │  │ Task ID → env_cfg_entry_pt   ││                                │
      │                 │  └───────────────────────────────│                                │
      │                 │                                  │                                │
      │  ┌─ import isaaclab_tasks ─┐                       │                                │
      │  │ isaaclab_tasks           │                      │                                │
      ├─────────────────────────────────────────────────────►│ gym.register()              │
      │  │                          │                      │ Add task to registry           │
      │  │                          │                      │                                │
      │  └──────────────────────────┘                      │                                │
      │                 │                                  │                                │
      │  ┌─ gym.make("Isaac-Cartpole-v0") ─┐               │                                │
      │  │ Task ID string                   │              │                                │
      │  │ Request environment              │              │                                │
      ├────────────────────────────────────────────────────►│ Registry lookup             │
      │  │                                  │              │ Find entry_point class        │
      │  │                                  │              │ Find env_cfg_entry_point      │
      │  └──────────────────────────────────┘              │                                │
      │                 │                                  │                                │
      │                 │  ┌──── Import class ────┐       │                                │
      │                 │  │ ManagerBasedRLEnv    │       │                                │
      │                 ├────────────────────────────────────►│ Retrieve class            │
      │                 │  │                      │       │ from isaaclab.envs             │
      │                 │  │                      │       │                                │
      │                 │  └──────────────────────┘       │                                │
      │                 │◄──────────────────────────────────┤ ManagerBasedRLEnv            │
      │                 │  Class object                    │ Class definition              │
      │                 │                                  │                               │
      │                 │  ┌──── Load config ────┐        │                                │
      │                 │  │ CartpoleEnvCfg      │        │                                │
      │                 ├────────────────────────────────────────────────────────────────────►│
      │                 │  │                     │        │                                │ Load config class
      │                 │  │                     │        │                                │ Scene, Actions
      │                 │  │                     │        │                                │ Obs, Rewards
      │                 │  │                     │        │                                │ Terms, Events
      │                 │  └─────────────────────┘        │                                │
      │                 │                                  │◄────────────────────────────────┤
      │                 │  cfg_instance                    │ CartpoleEnvCfg instance       │
      │                 │  Dataclass with:                 │ All MDP parameters loaded      │
      │                 │  • scene, actions, obs           │                                │
      │                 │  • rewards, terms, events        │                                │
      │                 │                                  │                                │
      │                 │  ┌──── Create env ────┐         │                                │
      │                 │  │ ManagerBasedRLEnv  │         │                                │
      │                 │  │ __init__(cfg=...)  │         │                                │
      │                 │  │ • SimulationContext│         │                                │
      │                 │  │ • InteractiveScene │         │                                │
      │                 │  │ • Managers (4)     │         │                                │
      │                 │  │ • Physics engine   │         │                                │
      │                 │  └────────────────────┘         │                                │
      │                 │                                  │                               │
      │  ┌─ env instance ──────────────────────────────────►│ ManagerBasedRLEnv            │
      │  │ Object: ManagerBasedRLEnv                   │  │ Fully initialized              │
      │  │ Ready for training                          │  │ Environment ready               │
      │  └──────────────────────────────────────────────┘  │                                │
      │                 │                                  │                                │
      │  ┌─ RlGamesVecEnvWrapper ─┐                        │                                │
      │  │ Wrap env for            │                        │                               │
      │  │ RL-Games compatibility  │                        │                               │
      │  └─────────────────────────┘                        │                               │
      │                 │                                  │                                │
      │  ┌─ runner.run() ──────┐                           │                                │
      │  │ Training loop        │                           │                               │
      │  │ NN policy → actions  │                           │                               │
      ├─────►env.step(action)─────────────────────────────────►│ Execute MDP step           │
      │  │   [REPEAT x1M]      │                           │ • ActionManager apply          │
      │  │                     │                           │ • Scene.step() physics         │
      │  │                     │                           │ • ObservationManager           │
      │  │                     │                           │ • RewardManager compute        │
      │  │                     │                           │ • TerminationManager check     │
      │  │                     │                           │                                │
      │  │◄──obs, reward,─────────────────────────────────────┤ Return gym.Env tuple  │
      │  │    done, info      │                           │ • obs: ndarray                  │
      │  │                    │                           │ • reward: float                 │
      │  └────────────────────┘                           │ • done: bool                    │
      │                 │                                  │ • info: dict                    │
      │                 │                                  │                                │
      ▼                 ▼                                  ▼                                ▼
   TRAINING         FACTORY PATTERN                   REGISTRY                         CONFIGURATION
   ────────────────────────────────────────────────────────────────────────────────────────────────────
   • Policy updates  • Dynamic imports                • gym.register()               • CartpoleSceneCfg
   • Loss backprop   • Class instantiation            • Entry point storage          • ActionsCfg
   • Reward accum    • Config loading                 • Config path storage          • ObservationsCfg
   • Gradient descent• Env instantiation              • Task ID mapping              • RewardsCfg
   • Weight optim    • Decouple training              • Enable decoupling            • TerminationsCfg
                     from tasks                                                       • EventCfg

════════════════════════════════════════════════════════════════════════════════════════════════════════════
DETAILED EXECUTION SEQUENCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① import isaaclab_tasks (train.py → __init__.py)
   OBJECT: isaaclab_tasks package | GOAL: Trigger gym.register() at import time
   
② gym.register(...) (__init__.py → GYMNASIUM)
   OBJECT: task_spec dict | GOAL: Add "Isaac-Cartpole-v0" to registry
   
③ gym.make("Isaac-Cartpole-v0") (train.py → GYMNASIUM)
   OBJECT: task_id string | GOAL: Request env creation via factory pattern
   
④ Registry lookup (GYMNASIUM → internal)
   OBJECT: entry_point + env_cfg_entry_point paths | GOAL: Find class and config locations
   
⑤ Import ManagerBasedRLEnv (GYMNASIUM → __init__.py)
   OBJECT: class path string | GOAL: Dynamically load environment class
   
⑥ Return class (GYMNASIUM ← __init__.py)
   OBJECT: ManagerBasedRLEnv class | GOAL: Provide class for instantiation
   
⑦ Import CartpoleEnvCfg (GYMNASIUM → cartpole_env_cfg.py)
   OBJECT: config path string | GOAL: Load complete MDP definition dataclass
   
⑧ Return config (GYMNASIUM ← cartpole_env_cfg.py)
   OBJECT: CartpoleEnvCfg instance | GOAL: Provide all MDP hyperparameters
   
⑨ Instantiate ManagerBasedRLEnv(cfg) (GYMNASIUM)
   OBJECT: cfg instance | GOAL: Build scene, physics, managers, initialize environment
   
⑩ Return env instance (GYMNASIUM → train.py)
   OBJECT: ManagerBasedRLEnv object | GOAL: Return ready-to-use environment
   
⑪ Wrap with RlGamesVecEnvWrapper (train.py)
   OBJECT: gym.Env → IVecEnv adapter | GOAL: Adapt interface to RL-Games
   
⑫ runner.run() → env.step(action) [LOOP x1M] (train.py → __init__.py)
   OBJECT: action array | GOAL: Step simulation, get obs/reward/done/info for policy update

═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
```
