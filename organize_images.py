import pandas as pd
import os
import shutil
from pathlib import Path
import zipfile
from tqdm import tqdm

def extract_and_organize_images(split_type='stratified_split', 
                               imgs_zip_path='data/imgs.zip',
                               extracted_dir='data/imgs',
                               organized_dir=None):
    """
    Extract images from zip file and organize them into train/val directory structure
    based on the CSV split files.
    
    Args:
        split_type: 'stratified_split' or 'subject_split'
        imgs_zip_path: Path to the imgs.zip file
        extracted_dir: Directory to extract images to
        organized_dir: Directory to organize images (if None, uses data/{split_type})
    """
    
    if organized_dir is None:
        organized_dir = f'data/{split_type}'
    
    # Load the split CSV files
    train_csv = f'{organized_dir}/train_list.csv'
    val_csv = f'{organized_dir}/val_list.csv'
    
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        print(f"Error: Split CSV files not found in {organized_dir}")
        print("Please run create_train_val_split.py first")
        return
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    print(f"Loading split data for {split_type}:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    
    # Extract images if not already extracted
    if not os.path.exists(extracted_dir):
        if not os.path.exists(imgs_zip_path):
            print(f"Error: {imgs_zip_path} not found")
            return
        
        print(f"Extracting {imgs_zip_path} to {extracted_dir}...")
        with zipfile.ZipFile(imgs_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
        print("Extraction completed!")
    else:
        print(f"Images already extracted to {extracted_dir}")
    
    # Create organized directory structure
    train_dir = os.path.join(organized_dir, 'train')
    val_dir = os.path.join(organized_dir, 'val')
    
    # Create class subdirectories
    classes = sorted(train_df['classname'].unique())
    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    
    # Function to copy images
    def copy_images(df, target_dir, set_name):
        print(f"\nCopying {set_name} images...")
        copied_count = 0
        error_count = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {set_name}"):
            class_name = row['classname']
            img_name = row['img']
            
            # Find source image (look in all subdirectories of extracted_dir)
            source_path = None
            for root, dirs, files in os.walk(extracted_dir):
                if img_name in files:
                    source_path = os.path.join(root, img_name)
                    break
            
            if source_path is None:
                print(f"Warning: Image {img_name} not found in {extracted_dir}")
                error_count += 1
                continue
            
            # Destination path
            dest_path = os.path.join(target_dir, class_name, img_name)
            
            try:
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {img_name}: {e}")
                error_count += 1
        
        print(f"{set_name} copying completed: {copied_count} images copied, {error_count} errors")
        return copied_count, error_count
    
    # Copy train images
    train_copied, train_errors = copy_images(train_df, train_dir, "Training")
    
    # Copy validation images
    val_copied, val_errors = copy_images(val_df, val_dir, "Validation")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"ORGANIZATION SUMMARY")
    print(f"="*60)
    print(f"Split type: {split_type}")
    print(f"Train images: {train_copied} copied ({train_errors} errors)")
    print(f"Validation images: {val_copied} copied ({val_errors} errors)")
    print(f"Total images: {train_copied + val_copied}")
    
    print(f"\nOrganized directory structure:")
    print(f"{organized_dir}/")
    print(f"├── train/")
    for class_name in classes:
        train_class_count = len([f for f in os.listdir(os.path.join(train_dir, class_name)) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"│   ├── {class_name}/ ({train_class_count} images)")
    print(f"└── val/")
    for class_name in classes:
        val_class_count = len([f for f in os.listdir(os.path.join(val_dir, class_name)) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"    ├── {class_name}/ ({val_class_count} images)")

def create_class_summary(split_type='stratified_split'):
    """Create a summary of class distributions in the organized dataset."""
    
    organized_dir = f'data/{split_type}'
    train_dir = os.path.join(organized_dir, 'train')
    val_dir = os.path.join(organized_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: Organized directories not found in {organized_dir}")
        return
    
    CLASS_DESCRIPTIONS = {
        'c0': 'safe driving',
        'c1': 'texting - right',
        'c2': 'talking on the phone - right', 
        'c3': 'texting - left',
        'c4': 'talking on the phone - left',
        'c5': 'operating the radio',
        'c6': 'drinking',
        'c7': 'reaching behind',
        'c8': 'hair and makeup',
        'c9': 'talking to passenger'
    }
    
    print(f"\nClass Summary for {split_type}:")
    print(f"{'Class':<5} {'Description':<25} {'Train':<8} {'Val':<6} {'Total':<8}")
    print("-" * 60)
    
    total_train = 0
    total_val = 0
    
    for class_name in sorted(os.listdir(train_dir)):
        if os.path.isdir(os.path.join(train_dir, class_name)):
            train_count = len([f for f in os.listdir(os.path.join(train_dir, class_name)) 
                             if f.endswith(('.jpg', '.jpeg', '.png'))])
            val_count = len([f for f in os.listdir(os.path.join(val_dir, class_name)) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            total_train += train_count
            total_val += val_count
            
            description = CLASS_DESCRIPTIONS.get(class_name, 'Unknown')
            print(f"{class_name:<5} {description:<25} {train_count:<8} {val_count:<6} {train_count + val_count:<8}")
    
    print("-" * 60)
    print(f"{'TOTAL':<31} {total_train:<8} {total_val:<6} {total_train + total_val:<8}")

if __name__ == "__main__":
    print("Image Organization Tool for Distracted Driver Classification")
    print("=" * 70)
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("Error: 'data' directory not found")
        print("Please ensure you're running this script from the correct directory")
        exit(1)
    
    # Check if imgs.zip exists
    if not os.path.exists('data/imgs.zip'):
        print("Warning: data/imgs.zip not found")
        print("Please ensure the images zip file is available")
        print("You can still run this script if images are already extracted")
    
    # Organize both splits
    print("\\n1. Organizing stratified split...")
    extract_and_organize_images(split_type='stratified_split')
    
    print("\\n" + "="*70)
    print("2. Organizing subject-based split...")
    extract_and_organize_images(split_type='subject_split')
    
    print("\\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # Create summaries
    create_class_summary('stratified_split')
    create_class_summary('subject_split')
    
    print("\\nImage organization completed!")
    print("You can now use these organized directories for training your model.")
