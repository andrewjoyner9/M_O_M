#!/usr/bin/env python3
"""
Real-time Path Visualization for DreamerV3 Training

This script shows the actual flight paths taken by the drone during training,
allowing you to see how navigation strategies evolve over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle
import os
from pathlib import Path
from collections import deque
import threading
import queue

class PathVisualizer:
    """Real-time visualization of drone flight paths"""
    
    def __init__(self, update_interval=2.0):
        self.update_interval = update_interval
        self.path_queue = queue.Queue()
        
        # Setup 3D plotting
        self.fig = plt.figure(figsize=(15, 10))
        
        # Create subplots
        self.ax_3d = self.fig.add_subplot(221, projection='3d')
        self.ax_top = self.fig.add_subplot(222)
        self.ax_success = self.fig.add_subplot(223)
        self.ax_efficiency = self.fig.add_subplot(224)
        
        self.fig.suptitle('DreamerV3 Navigation Path Analysis', fontsize=16)
        
        # Data storage
        self.recent_paths = deque(maxlen=20)
        self.success_history = deque(maxlen=100)
        self.efficiency_history = deque(maxlen=100)
        
        # Colors for different path types
        self.colors = {
            'success': 'green',
            'collision': 'red', 
            'timeout': 'orange',
            'current': 'blue'
        }
        
        self.running = False
        
    def setup_plots(self):
        """Initialize plot layouts"""
        # 3D path view
        self.ax_3d.set_title('3D Flight Paths (Recent Episodes)')
        self.ax_3d.set_xlabel('X Position')
        self.ax_3d.set_ylabel('Y Position')
        self.ax_3d.set_zlabel('Z Position')
        
        # Top-down view
        self.ax_top.set_title('Top-Down View (Recent Paths)')
        self.ax_top.set_xlabel('X Position')
        self.ax_top.set_ylabel('Y Position')
        self.ax_top.set_aspect('equal')
        self.ax_top.grid(True)
        
        # Success rate over time
        self.ax_success.set_title('Success Rate Over Time')
        self.ax_success.set_xlabel('Episode')
        self.ax_success.set_ylabel('Success Rate (%)')
        self.ax_success.grid(True)
        self.ax_success.set_ylim(0, 100)
        
        # Path efficiency
        self.ax_efficiency.set_title('Path Efficiency')
        self.ax_efficiency.set_xlabel('Episode')
        self.ax_efficiency.set_ylabel('Efficiency Score')
        self.ax_efficiency.grid(True)
        
        plt.tight_layout()
    
    def calculate_path_efficiency(self, path_data):
        """Calculate how efficient the path is (lower is better)"""
        if not path_data['path'] or len(path_data['path']) < 2:
            return 0
        
        path = np.array(path_data['path'])
        start = path[0]
        goal = path_data['goal']
        
        # Actual path length
        path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        
        # Direct distance
        direct_distance = np.linalg.norm(goal - start)
        
        # Efficiency: closer to 1.0 is better
        if direct_distance > 0:
            efficiency = direct_distance / path_length
        else:
            efficiency = 0
        
        return efficiency
    
    def plot_paths_3d(self, path_data_list):
        """Plot paths in 3D"""
        self.ax_3d.clear()
        self.ax_3d.set_title('3D Flight Paths (Recent Episodes)')
        self.ax_3d.set_xlabel('X Position')
        self.ax_3d.set_ylabel('Y Position')
        self.ax_3d.set_zlabel('Z Position')
        
        if not path_data_list:
            return
        
        # Collect all unique start/goal points and obstacles
        all_starts = []
        all_goals = []
        all_obstacles = []
        
        # Plot recent paths
        recent_episodes = path_data_list[-10:]  # Last 10 episodes
        for i, path_data in enumerate(recent_episodes):
            if not path_data.get('path'):
                continue
                
            path = np.array(path_data['path'])
            
            # Determine path color based on outcome
            if path_data.get('success', False):
                color = self.colors['success']
                alpha = 0.8
                linewidth = 3
            else:
                color = self.colors['collision']
                alpha = 0.6
                linewidth = 2
            
            # Plot path with episode-specific styling
            self.ax_3d.plot(path[:, 0], path[:, 1], path[:, 2], 
                           color=color, alpha=alpha, linewidth=linewidth,
                           label=f'Episode {len(path_data_list)-len(recent_episodes)+i+1} {"✓" if path_data.get("success", False) else "✗"}')
            
            # Collect start/goal for later plotting
            start = path[0]
            goal = path_data.get('goal', path[-1])  # Use final position if no goal specified
            all_starts.append(start)
            all_goals.append(goal)
            
            # Mark start and end of this specific path
            self.ax_3d.scatter(*start, color='darkblue', s=80, marker='o', alpha=0.8)
            self.ax_3d.scatter(*path[-1], color='darkred', s=80, marker='x', alpha=0.8)
            self.ax_3d.scatter(*goal, color='gold', s=120, marker='*', alpha=0.9)
            
            # Collect obstacles
            for obs_pos, obs_radius in path_data.get('obstacles', []):
                # Convert numpy array to tuple for hashing
                if isinstance(obs_pos, np.ndarray):
                    obs_pos_tuple = tuple(obs_pos)
                else:
                    obs_pos_tuple = tuple(obs_pos) if hasattr(obs_pos, '__iter__') else obs_pos
                all_obstacles.append((obs_pos_tuple, obs_radius))
        
        # Plot obstacles (avoid duplicates)
        unique_obstacles = []
        seen_obstacles = set()
        for obs_pos_tuple, obs_radius in all_obstacles:
            obstacle_key = (obs_pos_tuple, obs_radius)
            if obstacle_key not in seen_obstacles:
                seen_obstacles.add(obstacle_key)
                unique_obstacles.append((obs_pos_tuple, obs_radius))
        
        for obs_pos_tuple, obs_radius in unique_obstacles[:5]:  # Limit to 5 obstacles for clarity
            obs_pos = np.array(obs_pos_tuple)  # Convert back to array for plotting
            # Simple sphere representation  
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x = obs_radius * np.outer(np.cos(u), np.sin(v)) + obs_pos[0]
            y = obs_radius * np.outer(np.sin(u), np.sin(v)) + obs_pos[1]
            z = obs_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + obs_pos[2]
            self.ax_3d.plot_surface(x, y, z, alpha=0.2, color='red')
        
        # Add legend with custom labels
        legend_elements = [
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Start Points'),
            plt.Line2D([0], [0], color='red', marker='x', linewidth=0, markersize=8, label='End Points'),
            plt.Line2D([0], [0], color='gold', marker='*', linewidth=0, markersize=10, label='Goal Points'),
            plt.Line2D([0], [0], color=self.colors['success'], linewidth=3, label='Successful Paths'),
            plt.Line2D([0], [0], color=self.colors['collision'], linewidth=2, label='Failed Paths')
        ]
        self.ax_3d.legend(handles=legend_elements, loc='upper right')
    
    def plot_paths_2d(self, path_data_list):
        """Plot top-down view of paths"""
        self.ax_top.clear()
        self.ax_top.set_title('Top-Down View (Recent Paths)')
        self.ax_top.set_xlabel('X Position')
        self.ax_top.set_ylabel('Y Position')
        self.ax_top.set_aspect('equal')
        self.ax_top.grid(True)
        
        if not path_data_list:
            return
        
        # Plot recent paths
        recent_episodes = path_data_list[-10:]
        for i, path_data in enumerate(recent_episodes):
            if not path_data.get('path'):
                continue
                
            path = np.array(path_data['path'])
            
            # Color based on success
            if path_data.get('success', False):
                color = self.colors['success']
                alpha = 0.8
                linewidth = 3
            else:
                color = self.colors['collision']
                alpha = 0.6  
                linewidth = 2
            
            # Plot path (top-down)
            episode_num = len(path_data_list) - len(recent_episodes) + i + 1
            self.ax_top.plot(path[:, 0], path[:, 1], 
                           color=color, alpha=alpha, linewidth=linewidth,
                           label=f'Ep {episode_num} {"✓" if path_data.get("success", False) else "✗"}')
            
            # Plot start, end, and goal for this path
            start = path[0]
            end = path[-1]
            goal = path_data.get('goal', end)
            
            # Start point (blue circle)
            self.ax_top.scatter(start[0], start[1], color='darkblue', s=100, marker='o', 
                              edgecolors='white', linewidth=2, alpha=0.9)
            
            # End point (red X)
            self.ax_top.scatter(end[0], end[1], color='darkred', s=100, marker='x', 
                              alpha=0.9, linewidth=3)
            
            # Goal point (gold star)
            self.ax_top.scatter(goal[0], goal[1], color='gold', s=150, marker='*',
                              edgecolors='black', linewidth=1, alpha=0.9)
            
            # Show distance from end to goal
            distance = np.linalg.norm(end - goal)
            if distance > 0.5:  # Only show if significant distance
                self.ax_top.plot([end[0], goal[0]], [end[1], goal[1]], 
                               'k--', alpha=0.3, linewidth=1)
                mid_point = (end + goal) / 2
                self.ax_top.text(mid_point[0], mid_point[1], f'{distance:.1f}', 
                               ha='center', va='bottom', fontsize=8)
        
        # Plot obstacles (circles) - avoid duplicates
        all_obstacles = []
        for path_data in recent_episodes:
            for obs_pos, obs_radius in path_data.get('obstacles', []):
                # Convert numpy array to tuple for hashing
                if isinstance(obs_pos, np.ndarray):
                    obs_pos_tuple = tuple(obs_pos)
                else:
                    obs_pos_tuple = tuple(obs_pos) if hasattr(obs_pos, '__iter__') else obs_pos
                all_obstacles.append((obs_pos_tuple, obs_radius))
        
        unique_obstacles = []
        seen_obstacles = set()
        for obs_pos_tuple, obs_radius in all_obstacles:
            obstacle_key = (obs_pos_tuple, obs_radius)
            if obstacle_key not in seen_obstacles:
                seen_obstacles.add(obstacle_key)
                unique_obstacles.append((obs_pos_tuple, obs_radius))
        
        for obs_pos_tuple, obs_radius in unique_obstacles:
            obs_pos = np.array(obs_pos_tuple)  # Convert back to array
            circle = plt.Circle((obs_pos[0], obs_pos[1]), obs_radius, 
                              color='red', alpha=0.3, edgecolor='darkred')
            self.ax_top.add_patch(circle)
        
        # Custom legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', marker='o', linewidth=0, markersize=8, label='Start'),
            plt.Line2D([0], [0], color='red', marker='x', linewidth=0, markersize=8, label='End'),
            plt.Line2D([0], [0], color='gold', marker='*', linewidth=0, markersize=10, label='Goal'),
            plt.Line2D([0], [0], color=self.colors['success'], linewidth=3, label='Success'),
            plt.Line2D([0], [0], color=self.colors['collision'], linewidth=2, label='Failed')
        ]
        self.ax_top.legend(handles=legend_elements, loc='upper right')
    
    def plot_success_rate(self, path_data_list):
        """Plot success rate over time"""
        self.ax_success.clear()
        self.ax_success.set_title('Success Rate Over Time')
        self.ax_success.set_xlabel('Episode')
        self.ax_success.set_ylabel('Success Rate (%)')
        self.ax_success.grid(True)
        self.ax_success.set_ylim(0, 100)
        
        if len(path_data_list) < 5:
            return
        
        # Calculate rolling success rate
        window_size = 10
        success_rates = []
        episodes = []
        
        for i in range(window_size, len(path_data_list) + 1):
            window_data = path_data_list[i-window_size:i]
            successes = sum(1 for p in window_data if p.get('success', False))
            success_rate = (successes / window_size) * 100
            success_rates.append(success_rate)
            episodes.append(i)
        
        if success_rates:
            self.ax_success.plot(episodes, success_rates, 'b-', linewidth=2)
            
            # Add target line
            self.ax_success.axhline(y=95, color='r', linestyle='--', alpha=0.7, label='95% Target')
            self.ax_success.legend()
            
            # Show current success rate
            current_rate = success_rates[-1]
            self.ax_success.text(0.02, 0.98, f'Current: {current_rate:.1f}%', 
                               transform=self.ax_success.transAxes, 
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def plot_efficiency(self, path_data_list):
        """Plot path efficiency over time"""
        self.ax_efficiency.clear()
        self.ax_efficiency.set_title('Path Efficiency')
        self.ax_efficiency.set_xlabel('Episode')
        self.ax_efficiency.set_ylabel('Efficiency Score (Higher = Better)')
        self.ax_efficiency.grid(True)
        
        if len(path_data_list) < 5:
            return
        
        # Calculate efficiency for each episode
        efficiencies = []
        episodes = []
        
        for i, path_data in enumerate(path_data_list):
            efficiency = self.calculate_path_efficiency(path_data)
            if efficiency > 0:  # Only plot valid efficiencies
                efficiencies.append(efficiency)
                episodes.append(i + 1)
        
        if efficiencies:
            self.ax_efficiency.plot(episodes, efficiencies, 'g-', alpha=0.3)
            
            # Add rolling average
            if len(efficiencies) > 10:
                window_size = 10
                rolling_avg = []
                rolling_episodes = []
                
                for i in range(window_size, len(efficiencies) + 1):
                    avg_eff = np.mean(efficiencies[i-window_size:i])
                    rolling_avg.append(avg_eff)
                    rolling_episodes.append(episodes[i-1])
                
                self.ax_efficiency.plot(rolling_episodes, rolling_avg, 'g-', linewidth=2, label='Rolling Average')
                self.ax_efficiency.legend()
    
    def load_path_data(self):
        """Load path data from training environments"""
        # Look for path data files (including subdirectories)
        possible_paths = [
            "./logs_realistic_navigation/episode_paths.pkl",  # NEW: Realistic training data
            "./logs_realistic_phase1_easy/dreamerv3/episode_paths.pkl",
            "./logs_realistic_phase1_easy/episode_paths.pkl",
            "./logs_realistic_phase2_medium/dreamerv3/episode_paths.pkl",
            "./logs_realistic_phase2_medium/episode_paths.pkl", 
            "./logs_realistic_phase3_realistic/dreamerv3/episode_paths.pkl",
            "./logs_realistic_phase3_realistic/episode_paths.pkl",
            "./logs_realistic_phase4_expert/dreamerv3/episode_paths.pkl",
            "./logs_realistic_phase4_expert/episode_paths.pkl",
            "./logs_advanced/episode_paths.pkl"
        ]
        
        all_paths = []
        
        for path_file in possible_paths:
            path_file = Path(path_file)
            if path_file.exists():
                try:
                    with open(path_file, 'rb') as f:
                        paths = pickle.load(f)
                        all_paths.extend(paths)
                        print(f"✓ Loaded {len(paths)} paths from {path_file}")
                except Exception as e:
                    print(f"⚠️ Error loading {path_file}: {e}")
        
        return all_paths
    
    def update_visualization(self):
        """Update all plots with latest data"""
        # Try to load fresh path data
        path_data_list = self.load_path_data()
        
        if not path_data_list:
            # Create some dummy data for testing
            print("📊 No path data found, showing example visualization...")
            path_data_list = self.create_sample_data()
        
        # Update all plots
        self.plot_paths_3d(path_data_list)
        self.plot_paths_2d(path_data_list)
        self.plot_success_rate(path_data_list)
        self.plot_efficiency(path_data_list)
        
        # Update title with stats
        num_episodes = len(path_data_list)
        successes = sum(1 for p in path_data_list if p.get('success', False))
        success_rate = (successes / num_episodes * 100) if num_episodes > 0 else 0
        
        self.fig.suptitle(f'DreamerV3 Navigation Analysis - {num_episodes} Episodes - {success_rate:.1f}% Success Rate', 
                         fontsize=16)
        
        plt.tight_layout()
        plt.draw()
    
    def create_sample_data(self):
        """Create sample path data for demonstration"""
        sample_paths = []
        
        for i in range(20):
            # Create a curved path from start to goal
            start = np.array([-15 + np.random.randn(), -15 + np.random.randn(), 2])
            goal = np.array([15 + np.random.randn(), 15 + np.random.randn(), 2])
            
            # Generate path points
            num_points = np.random.randint(50, 200)
            t = np.linspace(0, 1, num_points)
            
            # Create curved path with some randomness
            path = []
            for ti in t:
                # Linear interpolation with noise
                pos = start + ti * (goal - start)
                # Add curve and noise
                pos[0] += np.sin(ti * np.pi * 2) * 3 + np.random.randn() * 0.5
                pos[1] += np.cos(ti * np.pi * 3) * 2 + np.random.randn() * 0.5
                pos[2] += np.sin(ti * np.pi * 4) * 1 + np.random.randn() * 0.2
                path.append(pos.copy())
            
            # Random success based on how close we get to goal
            final_pos = path[-1]
            distance_to_goal = np.linalg.norm(final_pos - goal)
            success = distance_to_goal < 3.0
            
            # Create obstacles
            obstacles = []
            for _ in range(5):
                obs_pos = np.random.uniform(-20, 20, 3)
                obs_radius = np.random.uniform(1, 3)
                obstacles.append((obs_pos, obs_radius))
            
            sample_paths.append({
                'path': path,
                'start': start,
                'goal': goal,
                'obstacles': obstacles,
                'success': success,
                'steps': len(path),
                'final_reward': np.random.uniform(-100, 300) if success else np.random.uniform(-500, 0)
            })
        
        return sample_paths
    
    def start_monitoring(self):
        """Start the visualization"""
        print("🎯 Starting DreamerV3 Path Visualization")
        print("=" * 50)
        
        self.running = True
        self.setup_plots()
        
        # Animation function
        def animate(frame):
            if self.running:
                self.update_visualization()
            return []
        
        # Start animation
        anim = animation.FuncAnimation(self.fig, animate, interval=int(self.update_interval * 1000), 
                                     blit=False, cache_frame_data=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n⏹️ Visualization stopped")
            self.running = False


def save_paths_from_env(env, save_dir="./logs_advanced"):
    """Save path data from environment for visualization"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Get episode paths from environment
    if hasattr(env, 'get_episode_paths'):
        episode_paths = env.get_episode_paths()
        
        if episode_paths:
            save_file = save_dir / "episode_paths.pkl"
            with open(save_file, 'wb') as f:
                pickle.dump(episode_paths, f)
            print(f"💾 Saved {len(episode_paths)} episode paths to {save_file}")
        else:
            print("⚠️ No episode paths found in environment")
    else:
        print("⚠️ Environment doesn't support path tracking")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize DreamerV3 flight paths")
    parser.add_argument('--interval', type=float, default=3.0,
                       help='Update interval in seconds')
    parser.add_argument('--demo', action='store_true',
                       help='Show demo with sample data')
    
    args = parser.parse_args()
    
    visualizer = PathVisualizer(update_interval=args.interval)
    
    if args.demo:
        print("🎮 Running demo mode with sample data")
    
    visualizer.start_monitoring()


if __name__ == "__main__":
    main()
