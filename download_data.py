import os
import shutil
from pathlib import Path
import kagglehub

def main():
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print("Path to dataset files:", path)
    # Manually define the path based on your system
    raw_data_path = Path.home() / ".cache" / "kagglehub" / "datasets" / "masoudnickparvar" / "brain-tumor-mri-dataset" / "versions" / "1" 

    print(f"🔍 Checking path: {raw_data_path}")
    if not raw_data_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found at: {raw_data_path}")

    # Define the destination inside your project
    dest_path = Path("data") / "Brain_Cancer"
    os.makedirs("data", exist_ok=True)

    if not dest_path.exists():
        print(f"📂 Copying dataset to: {dest_path}")
        shutil.copytree(raw_data_path, dest_path)
    else:
        print(f"✅ Dataset already exists at {dest_path}, skipping copy.")

    print("🎉 Dataset is ready in ./data/Brain_Cancer")

if __name__ == "__main__":
    main()