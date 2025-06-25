import pandas as pd
import numpy as np

# Define class mappings
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

def analyze_split(split_type):
    """Analyze a specific train/validation split."""
    
    print(f"\n{split_type.upper()} ANALYSIS")
    print("=" * 60)
    
    # Load the data
    train_df = pd.read_csv(f'data/{split_type}/train_list.csv')
    val_df = pd.read_csv(f'data/{split_type}/val_list.csv')
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Total samples: {len(train_df) + len(val_df)}")
    print(f"Train/Val ratio: {len(train_df)/len(val_df):.2f}")
    
    # Class distribution analysis
    print(f"\nClass distribution comparison:")
    print(f"{'Class':<5} {'Description':<25} {'Train':<8} {'Train%':<8} {'Val':<6} {'Val%':<8} {'Ratio':<8}")
    print("-" * 85)
    
    train_total = len(train_df)
    val_total = len(val_df)
    
    for class_name in sorted(train_df['classname'].unique()):
        train_count = len(train_df[train_df['classname'] == class_name])
        val_count = len(val_df[val_df['classname'] == class_name])
        
        train_pct = (train_count / train_total) * 100
        val_pct = (val_count / val_total) * 100
        ratio = train_pct / val_pct if val_pct > 0 else float('inf')
        
        description = CLASS_DESCRIPTIONS.get(class_name, 'Unknown')
        print(f"{class_name:<5} {description:<25} {train_count:<8} {train_pct:<7.2f}% {val_count:<6} {val_pct:<7.2f}% {ratio:<7.3f}")
    
    # Subject analysis
    if split_type == 'subject_split':
        train_subjects = set(train_df['subject'].unique())
        val_subjects = set(val_df['subject'].unique())
        overlap = train_subjects.intersection(val_subjects)
        
        print(f"\nSubject distribution:")
        print(f"  Train subjects: {len(train_subjects)} ({sorted(train_subjects)})")
        print(f"  Validation subjects: {len(val_subjects)} ({sorted(val_subjects)})")
        print(f"  Subject overlap: {len(overlap)}")
        
        if len(overlap) == 0:
            print("  ✅ No subject overlap - good for preventing data leakage")
        else:
            print(f"  ⚠️  Subject overlap detected: {sorted(overlap)}")
    
    # Balance analysis
    train_dist = train_df['classname'].value_counts(normalize=True)
    val_dist = val_df['classname'].value_counts(normalize=True)
    
    # Calculate KL divergence (measure of distribution difference)
    kl_div = np.sum(train_dist * np.log(train_dist / val_dist))
    
    print(f"\nBalance metrics:")
    print(f"  KL divergence (train vs val): {kl_div:.6f}")
    if kl_div < 0.01:
        print("  ✅ Excellent class balance maintained")
    elif kl_div < 0.05:
        print("  ✅ Good class balance maintained")  
    else:
        print("  ⚠️  Some class imbalance detected")
    
    return train_df, val_df

def compare_splits():
    """Compare both split types."""
    
    print("\nSPLIT COMPARISON")
    print("=" * 60)
    
    # Analyze both splits
    _, _ = analyze_split('stratified_split')
    _, _ = analyze_split('subject_split')
    
    print(f"\nRECOMMENDATIONS")
    print("=" * 60)
    print("1. STRATIFIED SPLIT:")
    print("   - Perfect class balance (identical distributions)")
    print("   - Best for: Initial model development and hyperparameter tuning")
    print("   - Caveat: Same subjects appear in train/val (potential overfitting)")
    
    print("\n2. SUBJECT-BASED SPLIT:")
    print("   - No subject overlap (prevents data leakage)")
    print("   - Best for: Final model evaluation and real-world deployment")
    print("   - Caveat: Slight class imbalance due to subject-based constraints")
    
    print("\n3. SUGGESTED WORKFLOW:")
    print("   a) Use stratified split for initial model development")
    print("   b) Use subject-based split for final evaluation")
    print("   c) Report results from subject-based split as final performance")

if __name__ == "__main__":
    print("Train/Validation Split Analysis")
    print("Distracted Driver MultiAction Classification")
    
    # Check if split files exist
    try:
        compare_splits()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run create_train_val_split.py first to create the splits")
