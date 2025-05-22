# Dqn_Implementation
A Deep Q-Network (DQN) agent implemented in PyTorch to solve the LunarLander  environment from OpenAI Gym. This project includes preprocessing of visual input, neural network design, experience replay, and gameplay video recording to monitor training progress.





# 🧠 Deep Q-Network (DQN) for LunarLander

This project implements a **Deep Q-Network (DQN)** to solve the `LunarLander` environment from OpenAI Gym using PyTorch. The notebook demonstrates the full pipeline from environment setup, preprocessing, network design, to training and video recording.

---

## 📌 Project Highlights

- **Environment**: `LunarLander` from OpenAI Gym
- **State Representation**: Stacked frames (4 grayscale images of size 84×84)
- **Action Space**: 4 discrete actions
- **Algorithm**: DQN with experience replay and target network
- **Visualization**: Recorded gameplay every 10 episodes

---

## 🧱 Code Components Breakdown

### 📦 Dependencies

The notebook installs and uses:
- `gym`, `box2d`, `numpy`, `torch`
- `cv2` (OpenCV for preprocessing)
- `matplotlib`, `moviepy`, `pygame` for display and visualization

---

### 🎮 Environment Setup

- The environment is created with video recording using `RecordVideo`.
- Random seeds are set for reproducibility.
- The agent plays in a visual mode with rendered frames.

---

### 🖼️ Preprocessing

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

### 📉 Frame Stacking

```python
def stack_frames(stacked_frames, frame, is_new_episode):
    ...
    return stacked_state, stacked_frames
```

- Maintains a stack of the last 4 frames to capture temporal information.
- Used as the state input for the neural network.

---

### 🧠 Neural Network Architecture

```python
class DQN(nn.Module):
    ...
```

- Three convolutional layers extract spatial features.
- Followed by two fully connected layers.
- Outputs Q-values for each action.

---

### 🔁 Training Setup

- Two networks: `main_dqn` and `target_dqn`
- Target network is updated periodically.
- Uses `MSELoss` and Adam optimizer.

---

### 🎥 Video Recording

Every 10 episodes, a video of the agent's gameplay is recorded and saved in the `videos/` folder.

---

## ⚙️ How to Run

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

## 🧠 Notes on DQN Algorithm

- Learns Q-values for state-action pairs using Bellman Equation.
- Uses replay buffer to break correlation between samples.
- Target network improves stability of training.

---

## 📂 Output

- Saved videos in `/videos/`
- Plots of rewards and loss during training

---


