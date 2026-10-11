import os
import shutil

base_dir = r"d:\Rock Classification\data\Dataset"

# Mapping to the correct superclass
mapping = {
    'Basalt': 'Igneous',
    'Granite': 'Igneous',
    'Sandstone': 'Sedimentary'
}

for rock, superclass in mapping.items():
    wrong_dir = os.path.join(base_dir, rock)
    correct_dir = os.path.join(base_dir, superclass, rock)
    
    if os.path.exists(wrong_dir) and os.path.isdir(wrong_dir):
        print(f"Moving images from {wrong_dir} to {correct_dir}")
        for f in os.listdir(wrong_dir):
            src = os.path.join(wrong_dir, f)
            dst = os.path.join(correct_dir, f)
            if os.path.isfile(src):
                shutil.move(src, dst)
        
        # remove the wrong dir if empty
        try:
            os.rmdir(wrong_dir)
            print(f"Removed {wrong_dir}")
        except Exception as e:
            print(f"Could not remove {wrong_dir}: {e}")
            
print("Fix complete.")
