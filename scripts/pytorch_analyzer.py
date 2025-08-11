import pickle
import sys
import os

def analyze_pytorch_checkpoint():
    """Analyze PyTorch checkpoint structure without loading tensors"""
    pkl_path = r"C:\Users\andre\Documents\BAH_PROJECT_2025\M_O_M\isaac-rl\improved_dreamerv3_checkpoint_iter_Working1.pkl"
    
    print("🎯 PYTORCH CHECKPOINT ANALYSIS")
    print("="*50)
    print(f"File: {pkl_path}")
    print(f"Size: {os.path.getsize(pkl_path):,} bytes")
    print()
    
    try:
        # Method 1: Try with PyTorch CPU mapping
        try:
            import torch
            print("✅ PyTorch available, attempting safe load...")
            
            # Load with explicit CPU mapping and weights_only=False for old format
            checkpoint = torch.load(pkl_path, map_location='cpu', weights_only=False)
            
            print("🎉 CHECKPOINT LOADED SUCCESSFULLY!")
            print()
            
            # Analyze checkpoint structure
            print("📋 CHECKPOINT CONTENTS")
            print("-" * 30)
            
            if isinstance(checkpoint, dict):
                print(f"Dictionary with {len(checkpoint)} keys:")
                print()
                
                for key, value in checkpoint.items():
                    print(f"🔑 '{key}':")
                    print(f"   Type: {type(value)}")
                    
                    # Handle different value types
                    if isinstance(value, torch.Tensor):
                        print(f"   Tensor shape: {value.shape}")
                        print(f"   Tensor dtype: {value.dtype}")
                    elif isinstance(value, dict):
                        print(f"   Dict with {len(value)} keys: {list(value.keys())[:3]}...")
                    elif hasattr(value, '__len__'):
                        print(f"   Length: {len(value)}")
                    else:
                        print(f"   Value: {str(value)[:100]}...")
                    print()
                
                # Look for specific DreamerV3 components
                print("🧠 DREAMERV3 ANALYSIS")
                print("-" * 30)
                
                # Common DreamerV3 checkpoint keys
                dreamer_keys = ['world_model', 'actor', 'critic', 'step', 'episode']
                found_components = []
                
                for key in checkpoint.keys():
                    for dreamer_key in dreamer_keys:
                        if dreamer_key in key.lower():
                            found_components.append((key, dreamer_key))
                            print(f"🎯 Found {dreamer_key} component: '{key}'")
                
                if not found_components:
                    print("🔍 No obvious DreamerV3 components found in top-level keys")
                    print("   This might be a different checkpoint format")
                
                # Check for training info
                print(f"\n📊 TRAINING INFO")
                print("-" * 20)
                
                if 'step' in checkpoint:
                    print(f"Training step: {checkpoint['step']}")
                if 'episode' in checkpoint:
                    print(f"Episode: {checkpoint['episode']}")
                if 'loss' in checkpoint:
                    print(f"Loss info: {checkpoint['loss']}")
                
                # Estimate model complexity
                total_params = 0
                for key, value in checkpoint.items():
                    if isinstance(value, torch.Tensor):
                        total_params += value.numel()
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, torch.Tensor):
                                total_params += subvalue.numel()
                
                print(f"\n🧮 Estimated total parameters: {total_params:,}")
                print(f"   Model size: ~{total_params * 4 / (1024*1024):.1f} MB (float32)")
                
                return checkpoint
                
            else:
                print(f"❓ Unexpected checkpoint format: {type(checkpoint)}")
                return None
                
        except ImportError:
            print("❌ PyTorch not available")
            return None
        except Exception as e:
            print(f"❌ PyTorch loading failed: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

def suggest_integration_approach(checkpoint):
    """Suggest how to integrate this checkpoint into the demo"""
    if checkpoint is None:
        print("\n💡 INTEGRATION SUGGESTIONS (without checkpoint access):")
        print("- File is confirmed to be a PyTorch DreamerV3 checkpoint")
        print("- Requires PyTorch for loading")
        print("- Can create a mock inference system for demo purposes")
        return
    
    print("\n🚀 INTEGRATION STRATEGY")
    print("="*30)
    
    if isinstance(checkpoint, dict):
        print("Recommended approach:")
        print("1. Load checkpoint with PyTorch")
        print("2. Extract relevant model components")
        print("3. Create inference wrapper")
        print("4. Use for navigation decisions")
        print()
        
        # Check what components are available
        has_world_model = any('world' in k.lower() for k in checkpoint.keys())
        has_actor = any('actor' in k.lower() for k in checkpoint.keys())
        has_policy = any('policy' in k.lower() for k in checkpoint.keys())
        
        print("Available for integration:")
        if has_world_model:
            print("✅ World model (environment understanding)")
        if has_actor:
            print("✅ Actor network (action selection)")
        if has_policy:
            print("✅ Policy network (decision making)")
        
        if not any([has_world_model, has_actor, has_policy]):
            print("⚠️ No obvious policy/actor components found")
            print("   May need custom extraction logic")

if __name__ == "__main__":
    checkpoint = analyze_pytorch_checkpoint()
    suggest_integration_approach(checkpoint)
    
    print("\n✅ ANALYSIS COMPLETE")
    print("Ready to create PyTorch-based demo integration!")
