"""
Isaac Sim interface for USD scene interaction and cube positioning
"""
from typing import Tuple, Optional
try:
    from pxr import UsdGeom, Gf
    import omni
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False
    print("Warning: Isaac Sim libraries not available. Running in mock mode.")


class IsaacSimInterface:
    """Interface for interacting with Isaac Sim USD scene and objects"""
    
    def __init__(self, cube_path: str = "/World/Cube"):
        self.cube_path = cube_path
        
        if ISAAC_SIM_AVAILABLE:
            self.stage = omni.usd.get_context().get_stage()
            self.cube_prim = self.stage.GetPrimAtPath(cube_path)
        else:
            self.stage = None
            self.cube_prim = None
            self._mock_position = Gf.Vec3f(0, 0, 0) if 'Gf' in globals() else (0, 0, 0)
    
    def get_cube_position(self) -> Tuple[float, float, float]:
        """Get the current position of the cube"""
        if not ISAAC_SIM_AVAILABLE:
            # Mock implementation
            pos = self._mock_position
            return (float(pos[0]), float(pos[1]), float(pos[2])) if hasattr(pos, '__getitem__') else (0, 0, 0)
        
        if not self.cube_prim:
            print(f"Error: Cube not found at {self.cube_path}")
            return (0, 0, 0)
        
        xform = UsdGeom.Xformable(self.cube_prim)
        translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        if translate_ops:
            pos = translate_ops[0].Get()
            return (float(pos[0]), float(pos[1]), float(pos[2]))
        return (0, 0, 0)

    def set_cube_position(self, x: float, y: float, z: float) -> bool:
        """Set the position of the cube"""
        if not ISAAC_SIM_AVAILABLE:
            # Mock implementation
            self._mock_position = (x, y, z)
            print(f"Mock: Cube moved to position: ({x}, {y}, {z})")
            return True
        
        if not self.cube_prim:
            print(f"Error: Cube not found at {self.cube_path}")
            return False
        
        xform = UsdGeom.Xformable(self.cube_prim)
        translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        
        if translate_ops:
            translate_ops[0].Set(Gf.Vec3f(x, y, z))
        else:
            xform.AddTranslateOp().Set(Gf.Vec3f(x, y, z))
        
        print(f"Cube moved to position: ({x}, {y}, {z})")
        return True
    
    def get_stage(self):
        """Get the USD stage"""
        return self.stage
    
    def is_cube_valid(self) -> bool:
        """Check if the cube primitive is valid"""
        if not ISAAC_SIM_AVAILABLE:
            return True  # Mock always valid
        return self.cube_prim is not None and self.cube_prim.IsValid()
    
    def update_cube_path(self, new_cube_path: str) -> bool:
        """Update the cube path and refresh the primitive reference"""
        self.cube_path = new_cube_path
        
        if ISAAC_SIM_AVAILABLE and self.stage:
            self.cube_prim = self.stage.GetPrimAtPath(new_cube_path)
            return self.is_cube_valid()
        return True  # Mock mode
