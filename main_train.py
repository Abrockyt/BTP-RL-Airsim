# ==============================================================================
# PROMPT FOR COPILOT:
# Main Training Loop for the MHA-PPO UAV Project.
# 1. Initialize the AirSimEnv.
# 2. Initialize the PPO_Agent.
# 3. Run for 1000 Episodes.
# 4. In every step, save the 'transition' (state, action, reward) to the agent's memory.
# 5. Every 2000 timesteps, call agent.update() to train the neural network.
# 6. Save the model weights every 50 episodes.
# ==============================================================================

from airsim_gym_wrapper import AirSimEnv
from mha_ppo_agent import PPO_Agent
import torch
import numpy as np
import os
from datetime import datetime

def train():
    """Main training loop for MHA-PPO UAV navigation"""
    
    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize Environment
    print("=" * 70)
    print("MHA-PPO UAV Training - Energy-Efficient Navigation in Turbulent Wind")
    print("=" * 70)
    print("\nInitializing AirSim Environment...")
    
    env = AirSimEnv(goal_position=[50, 50, -10])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State Dimension: {state_dim}")
    print(f"Action Dimension: {action_dim}")
    
    # Initialize PPO Agent with Multi-Head Attention
    print("\nInitializing MHA-PPO Agent...")
    agent = PPO_Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.0003,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=4
    )
    
    print("Agent initialized with Multi-Head Attention Actor Network")
    
    # Training statistics
    total_timesteps = 0
    best_reward = -float('inf')
    episode_rewards = []
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_log_{timestamp}.txt"
    
    with open(log_file, 'w') as f:
        f.write("Episode,Timesteps,Reward,Avg_Reward,Battery_Left,Distance_to_Goal,Done_Reason\n")
    
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    
    # Main Training Loop
    for episode in range(1, 1001):
        state = env.reset()
        ep_reward = 0
        ep_timesteps = 0
        
        for t in range(500):  # Max steps per episode
            total_timesteps += 1
            ep_timesteps += 1
            
            # Select Action using the MHA-PPO Brain
            action = agent.select_action(state)
            
            # Execute Action in AirSim
            next_state, reward, done, info = env.step(action)
            
            # Store reward and done flag (agent already stored state, action in select_action)
            agent.store_reward(reward, done)
            
            state = next_state
            ep_reward += reward
            
            # Update the Brain every 2000 timesteps
            if total_timesteps % 2000 == 0:
                print(f"\n[TRAINING] Updating policy at timestep {total_timesteps}...")
                agent.update()
                print("[TRAINING] Update complete!")
            
            if done:
                break
        
        # Episode statistics
        episode_rewards.append(ep_reward)
        avg_reward = np.mean(episode_rewards[-100:])  # Average over last 100 episodes
        
        # Console output
        print(f"\nEpisode {episode:4d} | "
              f"Timesteps: {ep_timesteps:3d} | "
              f"Reward: {ep_reward:8.2f} | "
              f"Avg(100): {avg_reward:8.2f}")
        print(f"  ├─ Battery: {info['battery_percent']:5.1f}% | "
              f"Distance: {info['distance_to_goal']:6.2f}m | "
              f"Done: {info['done_reason']}")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"{episode},{total_timesteps},{ep_reward:.2f},{avg_reward:.2f},"
                   f"{info['battery_percent']:.1f},{info['distance_to_goal']:.2f},"
                   f"{info['done_reason']}\n")
        
        # Save best model
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(f"models/mha_ppo_best.pth")
            print(f"  └─ ★ New best model saved! Reward: {best_reward:.2f}")
        
        # Save Model Checkpoint every 50 episodes
        if episode % 50 == 0:
            checkpoint_path = f"models/mha_ppo_episode_{episode}.pth"
            agent.save(checkpoint_path)
            print(f"\n{'='*70}")
            print(f"CHECKPOINT: Model saved at episode {episode}")
            print(f"  ├─ Total timesteps: {total_timesteps}")
            print(f"  ├─ Average reward (last 100): {avg_reward:.2f}")
            print(f"  └─ Best reward so far: {best_reward:.2f}")
            print(f"{'='*70}\n")
        
        # Performance summary every 100 episodes
        if episode % 100 == 0:
            success_count = sum(1 for i in range(max(0, episode-100), episode) 
                              if episode_rewards[i] > 50)  # Arbitrary success threshold
            print(f"\n{'='*70}")
            print(f"SUMMARY - Episodes {max(1, episode-99)}-{episode}")
            print(f"  ├─ Average Reward: {avg_reward:.2f}")
            print(f"  ├─ Best Reward: {max(episode_rewards[max(0, episode-100):episode]):.2f}")
            print(f"  ├─ Success Rate: {success_count}%")
            print(f"  └─ Total Timesteps: {total_timesteps}")
            print(f"{'='*70}\n")
    
    # Final cleanup
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Total Episodes: 1000")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Final Average Reward: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Best Reward Achieved: {best_reward:.2f}")
    print(f"\nModels saved in: ./models/")
    print(f"Logs saved in: {log_file}")
    
    # Save final model
    agent.save("models/mha_ppo_final.pth")
    print("Final model saved: models/mha_ppo_final.pth")
    
    # Close environment
    env.close()
    print("\nEnvironment closed. Training session ended.")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Progress has been saved in checkpoints.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
