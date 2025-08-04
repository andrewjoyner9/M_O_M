#!/usr/bin/env python3
"""
Setup script for Isaac Sim integration with the drone RL environment.
This script should be run from within Isaac Sim's Script Editor.
"""

def setup_isaac_sim_environment():
    """
    Set up the Isaac Sim environment for drone RL training.
    This creates the basic scene with lighting, physics, and ground plane.
    """
    
    try:
        # Isaac Sim imports
        from pxr import UsdGeom, Gf, UsdPhysics, UsdShade
        import omni
        from omni.isaac.core import World
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
        from omni.isaac.core.objects import GroundPlane
        import omni.physx as _physx
        
        print("Setting up Isaac Sim environment for drone RL...")
        
        # Get or create world
        world = World.instance()
        if world is None:
            world = World(stage_units_in_meters=1.0)
            print("Created new Isaac Sim world")
        else:
            print("Using existing Isaac Sim world")
        
        # Clear existing scene
        world.clear()
        
        # Add ground plane
        ground_plane = GroundPlane(
            prim_path="/World/GroundPlane",
            size=20.0,
            color=np.array([0.5, 0.5, 0.5])
        )
        world.scene.add(ground_plane)
        
        # Set up lighting
        stage = omni.usd.get_context().get_stage()
        
        # Add dome light for better visibility
        dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(1000)
        dome_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        
        # Add directional light (sun)
        sun_light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
        sun_light.CreateIntensityAttr(3000)
        sun_light.CreateAngleAttr(0.5)
        
        # Set light direction (coming from above and slightly angled)
        xformable = UsdGeom.Xformable(sun_light)
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))
        
        # Set up physics scene
        physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
        
        # Set physics time step for stability
        _physx.acquire_physx_interface().overwrite_gpu_setting(1)  # Enable GPU physics
        
        print("Isaac Sim environment setup completed!")
        print("Environment features:")
        print("- Ground plane (20x20 meters)")
        print("- Dome lighting for even illumination")
        print("- Directional sun light")
        print("- Physics scene with Earth gravity")
        print("- GPU physics enabled")
        
        return world
        
    except ImportError as e:
        print(f"Isaac Sim imports not available: {e}")
        print("This script must be run from within Isaac Sim")
        return None
    except Exception as e:
        print(f"Error setting up Isaac Sim environment: {e}")
        return None

def load_drone_rl_environment():
    """
    Load the drone RL environment into Isaac Sim.
    This should be called after setup_isaac_sim_environment().
    """
    
    try:
        import sys
        import os
        
        # Add the isaac-rl directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        # Import and create the Isaac Sim drone environment
        from isaac_sim_drone_env import IsaacSimDroneEnv
        
        print("Creating Isaac Sim drone environment...")
        env = IsaacSimDroneEnv(use_isaac_sim=True)
        
        print("Drone RL environment loaded successfully!")
        print("You can now train with:")
        print("  from train_isaac_sim import train_isaac_sim_drone")
        print("  train_isaac_sim_drone(use_isaac_sim=True)")
        
        return env
        
    except ImportError as e:
        print(f"Failed to import drone environment: {e}")
        return None
    except Exception as e:
        print(f"Error loading drone environment: {e}")
        return None

def run_isaac_sim_setup():
    """
    Complete setup process for Isaac Sim drone RL
    """
    
    print("=== Isaac Sim Drone RL Setup ===")
    
    # Step 1: Set up Isaac Sim environment
    world = setup_isaac_sim_environment()
    if world is None:
        print("Failed to set up Isaac Sim environment")
        return False
    
    # Step 2: Load drone RL environment
    env = load_drone_rl_environment()
    if env is None:
        print("Failed to load drone RL environment")
        return False
    
    print("\n=== Setup Complete! ===")
    print("Isaac Sim is ready for drone RL training.")
    print("\nNext steps:")
    print("1. Test the environment:")
    print("   obs, info = env.reset()")
    print("   action = env.action_space.sample()")
    print("   obs, reward, terminated, truncated, info = env.step(action)")
    print()
    print("2. Start training:")
    print("   exec(open('train_isaac_sim.py').read())")
    print()
    print("3. Or run quick demo:")
    print("   from train_isaac_sim import run_isaac_sim_demo")
    print("   run_isaac_sim_demo()")
    
    return True

def quick_test():
    """
    Quick test to verify everything is working
    """
    
    print("=== Quick Test ===")
    
    try:
        # Test Isaac Sim environment creation
        from isaac_sim_drone_env import IsaacSimDroneEnv
        
        env = IsaacSimDroneEnv(use_isaac_sim=True)
        print("✓ Environment created successfully")
        
        # Test reset
        obs, info = env.reset()
        print("✓ Environment reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Initial drone position: {obs[:3]}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("✓ Environment step successful")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.3f}")
        print(f"  New position: {obs[:3]}")
        
        # Clean up
        env.close()
        
        print("\n✓ All tests passed! Isaac Sim integration is working.")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

# Additional utility functions for Isaac Sim
def add_custom_assets():
    """
    Add custom drone model or other assets to the scene
    """
    try:
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Isaac Sim assets folder")
            return
        
        print(f"Isaac Sim assets available at: {assets_root_path}")
        
        # You can add custom USD assets here
        # Example: 
        # drone_asset_path = assets_root_path + "/Isaac/Robots/Quadrotor/quadrotor.usd"
        # add_reference_to_stage(usd_path=drone_asset_path, prim_path="/World/CustomDrone")
        
    except Exception as e:
        print(f"Error adding custom assets: {e}")

if __name__ == "__main__":
    # This will run when the script is executed
    print("Isaac Sim Drone RL Setup Script")
    print("===============================")
    print()
    print("To use this script in Isaac Sim:")
    print("1. Copy this file to your Isaac Sim project directory")
    print("2. In Isaac Sim Script Editor, run:")
    print("   exec(open('setup_isaac_sim.py').read())")
    print("3. Then call: run_isaac_sim_setup()")
    print()
    print("Or run individual functions:")
    print("- setup_isaac_sim_environment()")
    print("- load_drone_rl_environment()")
    print("- quick_test()")
