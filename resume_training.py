"""
Resume DRL training from checkpoint
Loads the last checkpoint and continues training
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from airsim_env import AirSimDroneEnv
import os
from datetime import datetime

# Custom callback to track training progress
class TrainingProgressCallback:
    def __init__(self):
        self.episode_rewards = []
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        
    def __call__(self, locals, globals):
        # Check if episode is done
        if locals['dones'][0]:
            self.episode_count += 1
            reward = locals['infos'][0].get('episode', {}).get('r', 0)
            self.episode_rewards.append(reward)
            
            # Track result type
            result = locals['infos'][0].get('result', 'unknown')
            if result == 'success':
                self.success_count += 1
            elif result == 'collision':
                self.collision_count += 1
            elif result == 'timeout':
                self.timeout_count += 1
            
            # Print progress every 10 episodes
            if self.episode_count % 10 == 0:
                avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
                print(f"\n📊 Episode {self.episode_count} | Avg Reward (last 10): {avg_reward:.2f} | "
                      f"Success: {self.success_count} | Collision: {self.collision_count} | Timeout: {self.timeout_count}")
        
        return True  # Continue training


def resume_training(checkpoint_path, additional_timesteps=50000, save_dir=None):
    """
    Resume training from a checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint .zip file
        additional_timesteps: Additional timesteps to train
        save_dir: Directory to save updated model (default: same as checkpoint)
    """
    print("=" * 70)
    print("🔄 RESUMING TRAINING FROM CHECKPOINT")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Additional Timesteps: {additional_timesteps:,}")
    print("=" * 70)
    
    # Create environment
    env = AirSimDroneEnv(
        goal_position=(30.0, 0.0),
        randomize_training=True
    )
    
    print("\n✓ Environment created")
    
    # Load the model from checkpoint
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env)
    
    print("✓ Model loaded successfully")
    
    # Setup save directory
    if save_dir is None:
        save_dir = os.path.dirname(checkpoint_path)
    
    print(f"\n📁 Models will be saved to: {save_dir}")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    progress_callback = TrainingProgressCallback()
    
    callbacks = CallbackList([checkpoint_callback])
    
    print("\n🚀 Resuming training...\n")
    print("-" * 70)
    
    try:
        # Continue training
        model.learn(
            total_timesteps=additional_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,  # Continue from current timestep count
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(save_dir, "smooth_drone_policy_final")
        model.save(final_path)
        print(f"\n✅ Training completed! Final model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user!")
        interrupted_path = os.path.join(save_dir, f"smooth_drone_policy_interrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        model.save(interrupted_path)
        print(f"✓ Model saved to: {interrupted_path}")
    
    finally:
        env.close()
        print("\n✓ Environment closed")
    
    return model, save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume PPO training from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file (e.g., checkpoint_40000_steps.zip)")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Additional timesteps to train (default: 50000)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save models (default: same as checkpoint)")
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        exit(1)
    
    # Resume training
    resume_training(
        checkpoint_path=args.checkpoint,
        additional_timesteps=args.timesteps,
        save_dir=args.save_dir
    )
