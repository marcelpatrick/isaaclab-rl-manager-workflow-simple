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


