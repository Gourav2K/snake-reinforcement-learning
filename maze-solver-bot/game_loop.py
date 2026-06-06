import os
from game_state_helpers import analyse_game_state, game_over, make_random_move, reward_function
from maze import create_maze
import numpy as np
from DQN import DQN
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
from experience_replay import ReplayMemory
from epsilon_choosing import select_action
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
from io import BytesIO
import imageio
from small_DQN import small_DQN

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend

def get_maze_state(maze, agent_position):
    state = np.copy(maze)
    state[agent_position] = 4  # Mark the agent's position with a unique value different from others
    return state

# Function to take a step in the environment
def step(maze, agent_position, action):
    size = maze.shape[0]
    action_space = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, left, down, up
    next_position = (agent_position[0] + action_space[action][0], agent_position[1] + action_space[action][1])
    if game_over(next_position, size):
        return agent_position, -10, True
    reward = reward_function(next_position, maze)

    # Add a small negative reward for each step to encourage efficiency
    step_penalty = -0.01
    reward += step_penalty

    done = agent_position == (size - 1, size - 1)
    return next_position, reward, done

# Function to render the maze with episode and step labels and return an in-memory image
def render_maze(maze, agent_position, size, episode, step):
    display = maze.copy()
    display[agent_position] = 'A'  # Mark the agent's position

    # Convert the display array to numeric values
    display_numeric = np.zeros(display.shape)
    display_numeric[display == 'S'] = 0
    display_numeric[display == 'E'] = 1
    display_numeric[display == 'T'] = 2
    display_numeric[display == 'F'] = 3
    display_numeric[display == 'A'] = 4

    fig, ax = plt.subplots()
    im = ax.imshow(display_numeric, cmap='viridis', interpolation='none')
    ax.text(0.5, 1.05, f'Episode: {episode}, Step: {step}', transform=ax.transAxes, ha="center")
    fig.canvas.draw()

    # Save the figure to a PIL image in memory
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

# Function to save the model state
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# Function to load the model state
def load_checkpoint(filename):
    return torch.load(filename)

def main(checkpoint_path=None):
    grid_size = 5
    seed = 173899
    num_traps = 6
    maze = create_maze(grid_size, num_traps, seed)
    print(maze)
    maze = np.array(maze)
    agent_position = (0, 0)
    print("Hi")

    maze_numeric = np.where(maze == 'S', 0, np.where(maze == 'E', 1, np.where(maze == 'T', 2, np.where(maze == 'F', 3, maze))))
    print(maze_numeric)
    initial_state = get_maze_state(maze_numeric, agent_position).astype(np.float32)

    # Parameters
    input_channels = 1  # Since we are using a single channel for the maze
    num_actions = 4  # Assuming 4 possible actions (up, down, left, right)
    batch_size = 64
    gamma = 0.9  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_min = 0.05
    epsilon_decay = 0.9975
    learning_rate = 0.001
    num_episodes = 1001
    target_update = 10
    checkpoint_dir = "checkpoints"

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Create the DQN model
    dqn_model = small_DQN(input_channels, num_actions, grid_size)
    target_model = small_DQN(input_channels, num_actions, grid_size)
    target_model.load_state_dict(dqn_model.state_dict())
    target_model.eval()

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(dqn_model.parameters(), lr=learning_rate)
    
    # Initialize replay memory
    Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))
    memory = ReplayMemory(2000)

    action_space = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, left, down, up

    # Store frames in memory
    frames = []

    # Load checkpoint if provided
    start_episode = 0
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)
        start_episode = checkpoint['episode'] + 1
        dqn_model.load_state_dict(checkpoint['dqn_model_state_dict'])
        target_model.load_state_dict(checkpoint['target_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epsilon = checkpoint['epsilon']
        total_reward = checkpoint['total_reward']
        print(f"Resuming training from episode {start_episode}")

    # Training loop
    writer = SummaryWriter()  # TensorBoard writer
    for episode in range(start_episode, start_episode+num_episodes):
        agent_position = (0, 0)  # Reset agent position to start for each episode
        state = get_maze_state(maze_numeric, agent_position).astype(np.float32)
        total_reward = 0
        done = False
        step_count = 0
        while not done:
            print("Episode - " + str(episode))
            action = select_action(state, epsilon, num_actions, dqn_model)
            print(action)
            next_position, reward, done = step(maze_numeric, agent_position, action)
            print("Next Position : " + str(next_position))
            print("Reward : " + str(reward))
            print("Done flag : " + str(done))
            next_state = get_maze_state(maze_numeric, next_position).astype(np.float32)
            
            memory.push(state, action, reward, next_state, done)
            
            state = next_state
            agent_position = next_position
            total_reward += reward
            step_count += 1

            if step_count>=200:
                done=True
            
            if len(memory) > batch_size:
                experiences = memory.sample(batch_size)
                batch = Experience(*zip(*experiences))
                states = torch.FloatTensor(batch.state).unsqueeze(1)
                actions = torch.LongTensor(batch.action)
                rewards = torch.FloatTensor(batch.reward)
                next_states = torch.FloatTensor(batch.next_state).unsqueeze(1)
                dones = torch.FloatTensor(batch.done)
                
                q_values = dqn_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_model(next_states).max(1)[0]
                expected_q_values = rewards + (gamma * next_q_values * (1 - dones))
                
                loss = criterion(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                # Store the frame
                if episode % 10 == 0:
                    frame = render_maze(maze, agent_position, grid_size, episode, step_count)
                    frames.append(frame)
    
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        if episode % target_update == 0:
            target_model.load_state_dict(dqn_model.state_dict())
        
        # Log metrics to TensorBoard
        writer.add_scalar('Total Reward', total_reward, episode)
        writer.add_scalar('Epsilon', epsilon, episode)
        writer.add_scalar('Step Count', step_count, episode)
        
        print(f"Episode {episode + 1}/{num_episodes+start_episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

        # Save checkpoint every 1000 episodes
        if episode % 100 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pth.tar")
            save_checkpoint({
                'episode': episode,
                'dqn_model_state_dict': dqn_model.state_dict(),
                'target_model_state_dict': target_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'total_reward': total_reward
            }, filename=checkpoint_path)

    writer.close()
    print("Training completed.")
    
    # Create GIF from frames
    frames[0].save('training_progress_small_grid.gif', save_all=True, append_images=frames[1:], duration=1000, loop=0)

    # Create video from frames
    video_frames = [np.array(frame) for frame in frames]
    imageio.mimsave('training_progress_small_grid.mp4', video_frames, fps=2)  # 2 frames per second for the video


if __name__ == '__main__':
    #checkpoint_path = "checkpoints/checkpoint_1500.pth.tar"  # Path to the checkpoint file
    #main(checkpoint_path)
    main()
