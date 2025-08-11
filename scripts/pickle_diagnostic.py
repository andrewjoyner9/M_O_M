import pickle
import sys
import os

def analyze_pickle_file():
    """Analyze the contents of the DreamerV3 pickle file"""
    pkl_path = r"C:\Users\andre\Documents\BAH_PROJECT_2025\M_O_M\isaac-rl\improved_dreamerv3_checkpoint_iter_Working1.pkl"
    
    print("🔍 DREAMERV3 PICKLE FILE DIAGNOSTIC")
    print("="*50)
    print(f"File: {pkl_path}")
    print()
    
    # Check if file exists
    if not os.path.exists(pkl_path):
        print("❌ File not found!")
        return
    
    # Get file size
    file_size = os.path.getsize(pkl_path)
    print(f"📦 File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    print()
    
    try:
        print("🔄 Loading pickle file...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print("✅ Successfully loaded!")
        print()
        
        # Analyze data structure
        print("📋 DATA STRUCTURE ANALYSIS")
        print("-" * 30)
        print(f"Root type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Dictionary with {len(data)} keys:")
            print()
            
            for i, (key, value) in enumerate(data.items()):
                print(f"🔑 Key {i+1}: '{key}'")
                print(f"   Type: {type(value)}")
                
                # Try to get more info about the value
                if hasattr(value, 'shape'):
                    print(f"   Shape: {value.shape}")
                elif hasattr(value, '__len__'):
                    try:
                        print(f"   Length: {len(value)}")
                    except:
                        pass
                
                # Show first few characters if it's a string or small object
                str_repr = str(value)
                if len(str_repr) < 200:
                    print(f"   Value: {str_repr}")
                else:
                    print(f"   Preview: {str_repr[:100]}...")
                
                print()
        
        elif isinstance(data, (list, tuple)):
            print(f"Collection with {len(data)} items:")
            for i, item in enumerate(data[:5]):  # Show first 5 items
                print(f"  Item {i}: {type(item)} - {str(item)[:50]}...")
            if len(data) > 5:
                print(f"  ... and {len(data) - 5} more items")
        
        else:
            print(f"Single object of type: {type(data)}")
            if hasattr(data, '__dict__'):
                print("Object attributes:")
                for attr in dir(data):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(data, attr)
                            print(f"  {attr}: {type(value)}")
                        except:
                            print(f"  {attr}: <unable to access>")
        
        # Look for common ML model components
        print("\n🧠 MODEL COMPONENT SEARCH")
        print("-" * 30)
        
        def search_for_keywords(obj, path=""):
            """Recursively search for ML-related keywords"""
            keywords = ['policy', 'network', 'weights', 'model', 'actor', 'critic', 
                       'encoder', 'decoder', 'dreamer', 'world_model', 'state', 'action']
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check if key contains any keywords
                    for keyword in keywords:
                        if keyword.lower() in key.lower():
                            print(f"🎯 Found '{keyword}' in: {current_path}")
                            print(f"   Type: {type(value)}")
                            if hasattr(value, 'shape'):
                                print(f"   Shape: {value.shape}")
                            break
                    
                    # Recurse into nested dictionaries (but limit depth)
                    if isinstance(value, dict) and len(path.split('.')) < 3:
                        search_for_keywords(value, current_path)
        
        search_for_keywords(data)
        
        print("\n✅ DIAGNOSTIC COMPLETE")
        print("="*50)
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading pickle file: {e}")
        print(f"Error type: {type(e)}")
        
        # Try to get more details about the error
        if "numpy" in str(e).lower():
            print("\n💡 NUMPY ISSUE DETECTED")
            print("The pickle file likely contains NumPy arrays.")
            print("This is common for ML models.")
            print("Isaac Sim may not have NumPy available by default.")
        
        return None

def simple_file_info():
    """Get basic file info without loading the pickle"""
    pkl_path = r"C:\Users\andre\Documents\BAH_PROJECT_2025\M_O_M\isaac-rl\improved_dreamerv3_checkpoint_iter_Working1.pkl"
    
    print("\n📊 BASIC FILE INFO")
    print("-" * 20)
    
    if os.path.exists(pkl_path):
        stat = os.stat(pkl_path)
        print(f"Size: {stat.st_size:,} bytes")
        print(f"Modified: {stat.st_mtime}")
        
        # Try to peek at the file header
        try:
            with open(pkl_path, 'rb') as f:
                header = f.read(50)
                print(f"Header bytes: {header[:20]}")
        except:
            pass
    else:
        print("File not found!")

if __name__ == "__main__":
    # Run diagnostics
    data = analyze_pickle_file()
    simple_file_info()
    
    print("\n🎯 NEXT STEPS:")
    print("Based on this analysis, we can create proper model integration!")
