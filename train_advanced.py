"""
Advanced DRL Training for Maximum Obstacle Avoidance
Trains for 500K+ timesteps with comprehensive scenarios for robust navigation
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback
from airsim_env import AirSimDroneEnv
import os
from datetime import datetime
import numpy as np


class AdvancedTrainingCallback(BaseCallback):
    """Advanced callback with detailed progress tracking and episode-based checkpoints"""
    def __init__(self, save_path, save_freq=10, verbose=0):
        super(AdvancedTrainingCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq  # Save every N episodes
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.total_distance_traveled = 0
        
    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.episode_count += 1
            reward = self.locals['infos'][0].get('episode', {}).get('r', 0)
            length = self.locals['infos'][0].get('episode', {}).get('l', 0)
            
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            
            result = self.locals['infos'][0].get('result', 'unknown')
            if result == 'success':
                self.success_count += 1
            elif result == 'collision':
                self.collision_count += 1
            elif result == 'timeout':
                self.timeout_count += 1
            
            # Save checkpoint every N episodes
            if self.episode_count % self.save_freq == 0:
                checkpoint_path = os.path.join(
                    self.save_path,
                    f"episode_{self.episode_count}_steps_{self.num_timesteps}.zip"
                )
                self.model.save(checkpoint_path)
                if self.verbose > 0:
                    print(f"\n💾 Checkpoint saved: episode_{self.episode_count}_steps_{self.num_timesteps}.zip")
            
            # Print detailed progress every 5 episodes
            if self.episode_count % 5 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else np.mean(self.episode_lengths)
                success_rate = (self.success_count / self.episode_count) * 100
                
                print(f"\n{'='*70}")
                print(f"📊 Episode {self.episode_count} Summary (Timestep: {self.num_timesteps:,})")
                print(f"{'='*70}")
                print(f"  Avg Reward (last 10): {avg_reward:.2f}")
                print(f"  Avg Length (last 10): {avg_length:.0f} steps")
                print(f"  Success Rate: {success_rate:.1f}% ({self.success_count}/{self.episode_count})")
                print(f"  Collisions: {self.collision_count} ({(self.collision_count/self.episode_count)*100:.1f}%)")
                print(f"  Timeouts: {self.timeout_count} ({(self.timeout_count/self.episode_count)*100:.1f}%)")
                print(f"{'='*70}\n")
        
        return True


def train_advanced_pilot(total_timesteps=500000, resume_from=None):
    """
    Train advanced navigation with comprehensive obstacle scenarios
    
    Args:
        total_timesteps: Number of timesteps to train (default: 500K)
        resume_from: Path to checkpoint to resume from (optional)
    """
    print("="*70)
    print("🚁 ADVANCED DRONE NAVIGATION TRAINING")
    print("="*70)
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Goal: Robust obstacle avoidance in complex environments")
    print(f"Curriculum Learning: Easy → Medium → Hard → Extreme")
    print("="*70)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"trained_models/advanced_pilot_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/tensorboard", exist_ok=True)
    
    print(f"\n📁 Models will be saved to: {save_dir}")
    
    # Create environment with randomized training
    env = AirSimDroneEnv(
        goal_position=(30.0, 0.0),
        randomize_training=True
    )
    
    print("\n✓ Environment created")
    
    # Create or load model
    if resume_from and os.path.exists(resume_from):
        print(f"\n📂 Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=env)
        print("✓ Model loaded")
    else:
        print("\n🤖 Creating new PPO model with optimized hyperparameters...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,  # Reduced from 2048 for faster updates
            batch_size=128,  # Increased from 64 for better GPU utilization
            n_epochs=5,  # Reduced from 10 for faster training
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=f"{save_dir}/tensorboard",
            verbose=1
        )
        print("✓ Model created")
    
    print("\n🎯 Hyperparameters (Speed Optimized):")
    print(f"   - Learning Rate: 3e-4")
    print(f"   - Batch Size: 128 (↑ faster processing)")
    print(f"   - N Steps: 1024 (↓ faster updates)")
    print(f"   - N Epochs: 5 (↓ faster training)")
    print(f"   - Entropy Coef: 0.01 (exploration)")
    print(f"   - Clip Range: 0.2")
    print(f"\n💾 Checkpoint Strategy:")
    print(f"   - Every 5 episodes (more frequent saves)")
    print(f"   - Format: episode_N_steps_M.zip")
    
    # Setup callbacks
    progress_callback = AdvancedTrainingCallback(
        save_path=save_dir,
        save_freq=5,  # Save every 5 episodes (more frequent to handle crashes)
        verbose=1
    )
    
    callbacks = CallbackList([progress_callback])
    
    print("\n🚀 Starting training...")
    print("-"*70)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=(resume_from is None)
        )
        
        # Save final model
        final_path = os.path.join(save_dir, "advanced_pilot_final")
        model.save(final_path)
        print(f"\n✅ Training completed!")
        print(f"📦 Final model saved: {final_path}")
        
        # Print final statistics
        print("\n" + "="*70)
        print("📈 TRAINING COMPLETE - FINAL STATISTICS")
        print("="*70)
        print(f"Total Episodes: {progress_callback.episode_count}")
        print(f"Success Rate: {(progress_callback.success_count/progress_callback.episode_count)*100:.1f}%")
        print(f"Collision Rate: {(progress_callback.collision_count/progress_callback.episode_count)*100:.1f}%")
        print(f"Timeout Rate: {(progress_callback.timeout_count/progress_callback.episode_count)*100:.1f}%")
        print(f"Average Reward: {np.mean(progress_callback.episode_rewards[-100:]):.2f}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user!")
        interrupted_path = os.path.join(save_dir, f"advanced_pilot_interrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        model.save(interrupted_path)
        print(f"✓ Model saved to: {interrupted_path}")
    
    finally:
        env.close()
        print("\n✓ Environment closed")
    
    return model, save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced PPO Training for Obstacle Avoidance")
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Total timesteps to train (default: 500000)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    print("\n🎯 Training Configuration:")
    print(f"   Target: {args.timesteps:,} timesteps")
    print(f"   Estimated Time: ~{(args.timesteps / 10000) * 15:.0f} minutes")
    print(f"   Resume: {args.resume if args.resume else 'New training'}")
    
    input("\nPress ENTER to start training...")
    
    train_advanced_pilot(
        total_timesteps=args.timesteps,
        resume_from=args.resume
    )
