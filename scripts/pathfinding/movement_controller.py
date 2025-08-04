"""
Movement controller for managing cube movement and keyboard input
"""
from typing import List, Tuple, Optional, Callable
import time

try:
    import omni.appwindow
    import carb
    ISAAC_SIM_INPUT_AVAILABLE = True
except ImportError:
    ISAAC_SIM_INPUT_AVAILABLE = False
    print("Warning: Isaac Sim input libraries not available. Keyboard controls disabled.")


class MovementController:
    """Handles movement execution and keyboard controls"""
    
    def __init__(self, isaac_sim_interface, pathfinding_core, obstacle_manager):
        self.isaac_sim_interface = isaac_sim_interface
        self.pathfinding_core = pathfinding_core
        self.obstacle_manager = obstacle_manager
        
        # Movement state
        self.current_path: List[Tuple[int, int, int]] = []
        self.current_step = 0
        self.is_moving = False
        self.movement_delay = 1.0
        
        # Keyboard controls
        self.keyboard_subscription = None
        self.input_interface = None
        self.keyboard = None
        
        # Callbacks
        self.on_movement_complete: Optional[Callable] = None
        self.on_step_complete: Optional[Callable] = None
    
    def setup_keyboard_controls(self) -> bool:
        """Set up keyboard controls for step-by-step movement"""
        if not ISAAC_SIM_INPUT_AVAILABLE:
            print("Keyboard controls not available - Isaac Sim input libraries missing")
            return False
            
        try:
            app_window = omni.appwindow.get_default_app_window()
            keyboard = app_window.get_keyboard()
            
            if keyboard:
                self.keyboard = keyboard
                self.input_interface = carb.input.acquire_input_interface()
                self.keyboard_subscription = self.input_interface.subscribe_to_keyboard_events(
                    keyboard, self.on_keyboard_event
                )
                print("Keyboard controls activated! Press ENTER to advance to next step.")
                return True
            else:
                print("Failed to get keyboard from app window")
                return False
        except Exception as e:
            print(f"Failed to setup keyboard controls: {e}")
            return False

    def cleanup_keyboard_controls(self) -> None:
        """Clean up keyboard controls"""
        if self.keyboard_subscription is not None and ISAAC_SIM_INPUT_AVAILABLE:
            try:
                if hasattr(self, 'input_interface') and self.input_interface:
                    if hasattr(self, 'keyboard') and self.keyboard:
                        self.input_interface.unsubscribe_to_keyboard_events(
                            self.keyboard, self.keyboard_subscription
                        )
                    else:
                        # Try to get keyboard again for cleanup
                        app_window = omni.appwindow.get_default_app_window()
                        keyboard = app_window.get_keyboard()
                        if keyboard:
                            self.input_interface.unsubscribe_to_keyboard_events(
                                keyboard, self.keyboard_subscription
                            )
                self.keyboard_subscription = None
                print("Keyboard controls deactivated")
            except Exception as e:
                print(f"Error cleaning up keyboard controls: {e}")

    def on_keyboard_event(self, event) -> bool:
        """Handle keyboard events for step-by-step movement"""
        if not ISAAC_SIM_INPUT_AVAILABLE:
            return False
            
        try:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if event.input == carb.input.KeyboardInput.ENTER and self.is_moving:
                    self.advance_to_next_step()
                    return True
        except Exception as e:
            print(f"Error handling keyboard event: {e}")
        return False

    def advance_to_next_step(self) -> None:
        """Move to the next step in the path"""
        if not self.is_moving or self.current_step >= len(self.current_path):
            return
        
        x, y, z = self.current_path[self.current_step]
        
        # Check if the next position is blocked by an obstacle
        if self.obstacle_manager.is_position_blocked(x, y, z):
            print(f"Warning: Next position ({x}, {y}, {z}) is blocked by an obstacle!")
            self._handle_blocked_path()
            return
        
        print(f"Step {self.current_step + 1}/{len(self.current_path)}: Moving to ({x}, {y}, {z})")
        self.isaac_sim_interface.set_cube_position(x, y, z)
        
        self.current_step += 1
        
        # Call step complete callback
        if self.on_step_complete:
            self.on_step_complete(self.current_step, len(self.current_path))
        
        if self.current_step >= len(self.current_path):
            self._complete_movement()
        else:
            print(f"Press ENTER to continue to step {self.current_step + 1}...")

    def _handle_blocked_path(self) -> None:
        """Handle when the path becomes blocked during movement"""
        print("Attempting to replan path...")
        
        # Get current position and remaining target
        current_pos = self.isaac_sim_interface.get_cube_position()
        current_coords = (int(current_pos[0]), int(current_pos[1]), int(current_pos[2]))
        
        # Find the final destination from the remaining path
        if self.current_path:
            final_destination = self.current_path[-1]
            
            # Update obstacles and try to find a new path
            self.obstacle_manager.detect_obstacles_from_scene()
            obstacles = self.obstacle_manager.get_all_obstacles()
            new_path = self.pathfinding_core.find_best_path(current_coords, final_destination, obstacles)
            
            if new_path:
                print("Successfully replanned path!")
                self.current_path = new_path
                self.current_step = 0
                # Continue with the new path
                self.advance_to_next_step()
            else:
                print("Failed to find alternative path. Stopping movement.")
                self.stop_movement()
        else:
            print("No remaining path to replan. Stopping movement.")
            self.stop_movement()

    def _complete_movement(self) -> None:
        """Complete the movement sequence"""
        print("Path movement completed!")
        self.is_moving = False
        self.cleanup_keyboard_controls()
        
        # Call completion callback
        if self.on_movement_complete:
            self.on_movement_complete()

    def move_along_path_interactive(self, path: List[Tuple[int, int, int]]) -> None:
        """Move the cube along the calculated path with keyboard input for each step"""
        if not path:
            print("No path to follow!")
            return
        
        print(f"Starting interactive movement along path with {len(path)} steps...")
        print("Press ENTER to advance to each step...")
        
        self.current_path = path
        self.current_step = 0
        self.is_moving = True
        
        # Set up keyboard controls
        if self.setup_keyboard_controls():
            # Move to first step
            self.advance_to_next_step()
        else:
            print("Failed to set up keyboard controls. Falling back to manual mode.")
            print("Call movement_controller.advance_to_next_step() manually to advance.")
            # Still set up the state for manual advancement
            self.advance_to_next_step()

    def stop_movement(self) -> None:
        """Stop the current movement and clean up"""
        self.is_moving = False
        self.cleanup_keyboard_controls()
        print("Movement stopped")

    def get_movement_status(self) -> str:
        """Get the current movement status"""
        if self.is_moving:
            return f"Moving: Step {self.current_step}/{len(self.current_path)}"
        else:
            return "Not moving"

    def set_movement_speed(self, delay_seconds: float) -> None:
        """Set the delay between movements (legacy - now uses keyboard input)"""
        self.movement_delay = max(0.1, delay_seconds)
        print(f"Movement delay set to {self.movement_delay} seconds (Note: Now using keyboard input instead of time delays)")

    def set_callbacks(self, on_movement_complete: Optional[Callable] = None, 
                     on_step_complete: Optional[Callable] = None) -> None:
        """Set callback functions for movement events"""
        self.on_movement_complete = on_movement_complete
        self.on_step_complete = on_step_complete
