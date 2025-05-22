
##  Q-Table Agent: Feature-Based Discretization with Vision

We explored multiple versions of Q-table implementations, experimenting with different feature extraction pipelines, grid resolutions, and reward shaping strategies. Among these, we selected this **image-based Q-learning agent** because it consistently achieved better learning performance in simulation.

>  **Best performing version among all Q-table experiments**
>  **Learns purely from rendered frames**
>  **Discrete state representation via visual binning**
>  **Custom reward shaping for stability, alignment, and fuel efficiency**



####  **Image-Based State Encoding**

* **Grayscale Downsampling:** Captures a compact visual summary of the environment at `(84×84)` resolution.
* **Grid Feature Extraction:** Image split into a coarse `4×4` grid. Each cell is binarized based on average brightness.
* **Lander and Pad Localization:** Positions of lander and landing pad are discretized into bins (0–4), forming a compact symbolic state.

####  **Q-Table Discretization**

* **State Representation:** Each state is a tuple of 16 grid features + 3 bins (lander\_x\_bin, lander\_y\_bin, pad\_x\_bin).
* **Action Selection:** Uses ε-greedy strategy with decaying exploration.

####  **Custom Reward Function**

* Encourages:

  * Reducing distance to pad
  * Smooth vertical descent
  * Upright orientation
  * Lower fuel consumption
* Penalizes:

  * Excessive tilt and high horizontal speed
  * Long episode durations
  * Crashes or unstable flights

####  **Q-Table Update**

Classic Q-learning update rule:

```
Q(s, a) ← Q(s, a) + α [r + γ max(Q(s', ·)) − Q(s, a)]
```

Q-values are stored as tensors for GPU compatibility when available.

---

###  Training Setup

* **Environment:** `LunarLander-v3` (RGB frame input)
* **Training Episodes:** 2000
* **Exploration Schedule:**

  * Full exploration for 1300 episodes
  * Linear decay after that down to ε = 0.01

####  Decay Logic

```python
if current_episode >= exploration_threshold:
    epsilon = max(epsilon_min, epsilon - decay_rate)
```

---

###  Evaluation & Saving

* **Evaluation Method:** Runs the final policy over multiple test episodes using greedy action selection.
* **Saving/Loading:** Q-table is serialized with `pickle` for persistence and future evaluation.

---

###  Visualization Tools

* **Reward Plot:** Tracks total rewards over training episodes with a moving average curve.
* **Epsilon Plot:** Shows decay of exploration over time.
* **Episode Display:** Final training episode can be rendered as a video or GIF (Colab-compatible).

---

This agent demonstrates the potential of vision-based discrete RL when carefully crafted features and reward functions are used. Despite the limitations of tabular methods in large spaces, this implementation performs well due to intelligent feature design and reward tuning.

---

# Dqn_Implementation
A Deep Q-Network (DQN) agent implemented in PyTorch to solve the LunarLander  environment from OpenAI Gym. This project includes preprocessing of visual input, neural network design, experience replay, and gameplay video recording to monitor training progress.





#  Deep Q-Network (DQN) for LunarLander

This project implements a **Deep Q-Network (DQN)** to solve the `LunarLander` environment from OpenAI Gym using PyTorch. The notebook demonstrates the full pipeline from environment setup, preprocessing, network design, to training and video recording.

---

##  Project Highlights

- **Environment**: `LunarLander` from OpenAI Gym
- **State Representation**: Stacked frames (4 grayscale images of size 84×84)
- **Action Space**: 4 discrete actions
- **Algorithm**: DQN with experience replay and target network
- **Visualization**: Recorded gameplay every 10 episodes

---

##  Code Components Breakdown

###  Dependencies

The notebook installs and uses:
- `gym`, `box2d`, `numpy`, `torch`
- `cv2` (OpenCV for preprocessing)
- `matplotlib`, `moviepy`, `pygame` for display and visualization

---

###  Environment Setup

- The environment is created with video recording using `RecordVideo`.
- Random seeds are set for reproducibility.
- The agent plays in a visual mode with rendered frames.

---

###  Preprocessing

```python
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    normalized = resized.astype(np.float32) / 255.0
    return normalized
```

- Converts RGB frames to grayscale.
- Resizes frames to 84×84 pixels.
- Normalizes pixel values to [0, 1].

---

###  Frame Stacking

```python
def stack_frames(stacked_frames, frame, is_new_episode):
    ...
    return stacked_state, stacked_frames
```

- Maintains a stack of the last 4 frames to capture temporal information.
- Used as the state input for the neural network.

---

###  Neural Network Architecture

```python
class DQN(nn.Module):
    ...
```

- Three convolutional layers extract spatial features.
- Followed by two fully connected layers.
- Outputs Q-values for each action.

---

###  Training Setup

- Two networks: `main_dqn` and `target_dqn`
- Target network is updated periodically.
- Uses `MSELoss` and Adam optimizer.

---

###  Video Recording

Every 10 episodes, a video of the agent's gameplay is recorded and saved in the `videos/` folder.

---

##  How to Run

1. Install Jupyter and dependencies:

```bash
pip install notebook gym[box2d] torch opencv-python matplotlib moviepy pygame
```

2. Run the notebook:

```bash
jupyter notebook DQNCODE.ipynb
```

3. Make sure the notebook is in the same directory where `videos/` will be created.

---

##  Notes on DQN Algorithm

- Learns Q-values for state-action pairs using Bellman Equation.
- Uses replay buffer to break correlation between samples.
- Target network improves stability of training.

---

##  Output

- Saved videos in `/videos/`
- Plots of rewards and loss during training


#  PPO LunarLander Agent 

 A powerful **Proximal Policy Optimization (PPO)** agent trained to master the [OpenAI Gym LunarLander-v3](https://gym.openai.com/envs/LunarLander-v3/) environment using **deep reinforcement learning** and **frame-stacked image inputs**.

##  Architecture

The Agent uses a **PPO (Proximal Policy Optimization)** algorithm with **frame-stacked grayscale images** (4x84x84 input) and a **shared CNN backbone** that feeds into separate **actor and critic heads**.

###  Neural Network:

* **Input**: 4 stacked preprocessed grayscale frames (84×84)
* **CNN Backbone**:

  * `Conv2D(32 filters, 8x8 kernel, stride=4)`
  * `Conv2D(64 filters, 4x4 kernel, stride=2)`
  * `Conv2D(64 filters, 3x3 kernel, stride=1)`
* **Actor Head**:

  * `FC(512) → ReLU → FC(4)`
* **Critic Head**:

  * `FC(512) → ReLU → FC(1)`

---

##  Key Features

| Component         | Description                                                   |
| ----------------- | ------------------------------------------------------------- |
|  FrameStacking    | Captures temporal info by stacking 4 recent grayscale frames  |
|  PPO Algorithm    | Actor-Critic method with clipped objective for stable updates |
|  GAE              | Generalized Advantage Estimation for reduced variance         |
|  Entropy Bonus    | Encourages exploration by penalizing deterministic policies   |
|  Checkpoints      | Save/load model for inference and video rendering             |
|  Video Support    | Automatically records and generates `.mp4` + `.gif` episodes  |

---

##  How PPO Works (Simplified)

1. **Interact** with the environment, collect batches of state-action-reward data.
2. **Estimate advantages** using rewards and value function (via GAE).
3. **Update the policy**:

   * Maximize: clipped surrogate objective
   * Minimize: value loss + entropy bonus
4. **Repeat**, using experience batches every `update_interval` steps.

---

##  Training Configuration

```python
agent = PPOAgent(
    input_channels=4,
    num_actions=env.action_space.n,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    policy_clip=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    epochs=4,
    max_grad_norm=0.5
)
```

---

##  Training Loop

```python
train_ppo_agent(
    env_name='LunarLander-v2',
    num_episodes=1000,
    max_steps=1000,
    update_interval=2048,
    render_interval=100,
    save_interval=200
)
```

---

##  Visualizing Performance

```python
# Record and show a full episode
record_and_display_final_episode(agent, env_name='LunarLander-v2')
```

>  Produces both `.mp4` and `.gif` formats for easy visualization.




##  Results

| Metric         | Value (example)   |
| -------------- | ----------------- |
|  Avg. Reward   | \~240             |
|  Episodes      | 1000              |
|  Algorithm     | PPO               |
|  Time          | \~1.5 hours (GPU) |

---

##  Requirements

* `torch`
* `gym`
* `opencv-python`
* `imageio`
* `matplotlib`
* `numpy`
* `PIL`

Install all with:

```bash
pip install torch gym opencv-python imageio matplotlib numpy pillow
```

---


##  Credits

Final project for AI-based RL agent in simulation.

Inspired by:

* OpenAI Gym
* SpinningUp PPO
* DeepMind Atari strategies



