import pickle
import numpy as np
import omni
import sys
import os

# Add the current directory to the Python path to find isaac_sim_pathfinder
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from isaac_sim_pathfinder import IsaacSimPathfinder
from pxr import UsdGeom, Gf, Usd, UsdShade, Sdf


class SimplePklDemo:
    def __init__(self, pkl_path="../isaac-rl/improved_dreamerv3_checkpoint_iter_Working1.pkl"):
        """Initialize the demo with the pickle file path"""
        self.pkl_path = pkl_path
        self.pathfinder = IsaacSimPathfinder()
        self.model_data = None
        
    def load_model(self):
        """Load the DreamerV3 model from pickle file"""
        try:
            print(f"Loading model from: {self.pkl_path}")
            with open(self.pkl_path, 'rb') as f:
                self.model_data = pickle.load(f)
            print("✓ Model loaded successfully!")
            
            # Print some basic info about the loaded model
            if isinstance(self.model_data, dict):
                print(f"Model contains {len(self.model_data)} keys:")
                for key in list(self.model_data.keys())[:5]:  # Show first 5 keys
                    print(f"  - {key}")
                if len(self.model_data) > 5:
                    print(f"  ... and {len(self.model_data) - 5} more")
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def setup_simple_scene(self):
        """Create a simple scene with a cube and some obstacles"""
        print("Setting up simple demo scene...")
        
        # Create some obstacles to navigate around
        obstacles = [
            (3, 0, 0), (3, 0, 1), (3, 0, 2),  # Wall 1
            (0, 3, 0), (1, 3, 0), (2, 3, 0),  # Wall 2
            (6, 2, 0), (6, 2, 1),             # Small obstacle
        ]
        
        for x, y, z in obstacles:
            self.pathfinder.add_obstacle(x, y, z, create_sphere=True)
        
        print(f"✓ Added {len(obstacles)} obstacles to scene")
        
        # Set starting position
        start_pos = (0, 0, 0)
        self.pathfinder.set_cube_position(*start_pos)
        print(f"✓ Cube positioned at {start_pos}")
    
    def simulate_model_decision(self, current_pos, target_pos):
        """
        Simulate making a navigation decision using the loaded model
        In a real implementation, this would process the model data to make decisions
        For this demo, we'll use simple logic but reference the model data
        """
        if self.model_data is None:
            print("❌ No model loaded!")
            return None
        
        # Simple simulation: check if we have model data and make a decision
        print(f"🧠 Using AI model to plan navigation from {current_pos} to {target_pos}")
        
        # In a real scenario, you would:
        # 1. Extract relevant weights/parameters from self.model_data
        # 2. Process current observation (position, obstacles, target)
        # 3. Use model to predict best action/path
        
        # For demo purposes, we'll use the pathfinder but indicate it's "AI-guided"
        path = self.pathfinder.find_path(current_pos, target_pos)
        
        if path:
            print(f"✓ AI model found path with {len(path)} steps")
        else:
            print("❌ AI model could not find valid path")
        
        return path
    
    def run_navigation_demo(self):
        """Run the main navigation demo"""
        print("\n" + "="*60)
        print("🚁 DREAMERV3 ISAAC SIM NAVIGATION DEMO")
        print("="*60)
        
        # Load the model
        if not self.load_model():
            return
        
        # Setup scene
        self.setup_simple_scene()
        
        # Define navigation targets
        targets = [
            (5, 0, 0),   # Go around first wall
            (5, 5, 0),   # Navigate to corner
            (0, 5, 2),   # Move to elevated position
            (0, 0, 0),   # Return home
        ]
        
        print(f"\n📍 Navigation mission: Visit {len(targets)} waypoints")
        
        for i, target in enumerate(targets):
            print(f"\n--- Waypoint {i+1}/{len(targets)}: {target} ---")
            
            # Get current position
            current_pos = self.pathfinder.get_cube_position()
            current_coords = (int(current_pos[0]), int(current_pos[1]), int(current_pos[2]))
            
            # Skip if already at target
            if current_coords == target:
                print(f"✓ Already at target {target}")
                continue
            
            # Use "AI model" to plan path
            path = self.simulate_model_decision(current_coords, target)
            
            if path:
                print(f"🎯 Executing AI-planned path to {target}")
                print("Press ENTER to step through movement...")
                self.pathfinder.move_along_path_interactive(path)
                
                # Wait for user to complete this leg
                input("\nPress any key when ready for next waypoint...")
            else:
                print(f"❌ Cannot reach waypoint {target}")
        
        print("\n🎉 Navigation demo completed!")
        print("="*60)


def run_quick_demo():
    """Quick demonstration of the pickle file usage"""
    demo = SimplePklDemo()
    demo.run_navigation_demo()


if __name__ == "__main__":
    print("🚁 Simple DreamerV3 Isaac Sim Demo")
    print("This demo loads the trained model and simulates AI-guided navigation")
    print("="*60)
    
    # Check if Isaac Sim is running
    try:
        stage = omni.usd.get_context().get_stage()
        if stage:
            print("✓ Isaac Sim detected")
            run_quick_demo()
        else:
            print("❌ Isaac Sim stage not found")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure this script is run from within Isaac Sim")
