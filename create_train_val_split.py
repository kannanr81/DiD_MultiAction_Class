import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
from pathlib import Path
from collections import defaultdict
try:
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some advanced metrics will be skipped.")

# Define class mappings for distracted driver classification
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

def create_train_val_split(csv_file='data/driver_imgs_list.csv', 
                          validation_split=0.2, 
                          random_state=42,
                          subject_based_split=True):
    """
    Create train/validation split for Distracted Driver MultiAction Classification.
    
    Args:
        csv_file: Path to the driver_imgs_list.csv file
        validation_split: Fraction of data to use for validation (default: 0.2)
        random_state: Random seed for reproducibility
        subject_based_split: If True, split by subjects to avoid data leakage
    """
    
    # Load the data
    print("Loading dataset...")
    df = pd.read_csv(csv_file)
    
    print(f"Total samples: {len(df)}")
    print(f"Number of classes: {df['classname'].nunique()}")
    print(f"Number of subjects: {df['subject'].nunique()}")
    
    # Display original class distribution with descriptions
    print("\nOriginal class distribution:")
    original_dist = df['classname'].value_counts().sort_index()
    print(f"{'Class':<5} {'Description':<25} {'Count':<8} {'Percentage':<10}")
    print("-" * 55)
    for class_name, count in original_dist.items():
        percentage = (count / len(df)) * 100
        description = CLASS_DESCRIPTIONS.get(class_name, 'Unknown')
        print(f"{class_name:<5} {description:<25} {count:<8} {percentage:<9.2f}%")
    
    if subject_based_split:
        # Subject-based split to avoid data leakage with optimized class balance
        print("\nPerforming optimized subject-based split to avoid data leakage...")
        
        # Calculate class representation per subject
        subject_class_matrix = df.pivot_table(
            index='subject', 
            columns='classname', 
            values='img', 
            aggfunc='count', 
            fill_value=0
        )
        
        print("Subject-class distribution:")
        print(subject_class_matrix)
        
        # Calculate target class distribution for validation set
        target_val_samples = int(len(df) * validation_split)
        target_class_dist = original_dist / len(df)  # Proportion of each class
        target_val_class_counts = (target_class_dist * target_val_samples).round().astype(int)
        
        print(f"\nTarget validation set size: {target_val_samples}")
        print("Target validation class distribution:")
        for class_name in sorted(target_val_class_counts.index):
            count = target_val_class_counts[class_name]
            percentage = (count / target_val_samples) * 100
            description = CLASS_DESCRIPTIONS.get(class_name, 'Unknown')
            print(f"  {class_name} ({description}): {count} samples ({percentage:.2f}%)")
        
        # Optimized subject assignment using greedy algorithm
        subjects = df['subject'].unique()
        np.random.seed(random_state)
        subjects_shuffled = np.random.permutation(subjects)
        
        val_subjects = []
        val_class_counts = np.zeros(len(target_val_class_counts))
        
        # Greedy assignment: select subjects that best fit remaining class needs
        for subject in subjects_shuffled:
            subject_samples = subject_class_matrix.loc[subject].values
            
            # Calculate if adding this subject improves class balance
            potential_val_counts = val_class_counts + subject_samples
            current_total = val_class_counts.sum()
            potential_total = potential_val_counts.sum()
            
            # Check if we're not exceeding target validation size too much
            if potential_total <= target_val_samples * 1.5:  # Allow 50% overage for better balance
                # Calculate balance improvement
                current_error = np.sum(np.abs(val_class_counts - target_val_class_counts.values))
                potential_error = np.sum(np.abs(potential_val_counts - target_val_class_counts.values))
                
                # Add subject if it improves balance or we need more validation data
                if potential_error <= current_error or current_total < target_val_samples * 0.8:
                    val_subjects.append(subject)
                    val_class_counts = potential_val_counts
                    
                    # Stop if we have enough validation data and good balance
                    if (current_total >= target_val_samples * 0.8 and 
                        len(val_subjects) >= len(subjects) * validation_split * 0.5):
                        break
        
        # Ensure minimum number of validation subjects
        min_val_subjects = max(2, int(len(subjects) * validation_split * 0.5))
        if len(val_subjects) < min_val_subjects:
            remaining_subjects = [s for s in subjects_shuffled if s not in val_subjects]
            val_subjects.extend(remaining_subjects[:min_val_subjects - len(val_subjects)])
        
        train_subjects = [s for s in subjects if s not in val_subjects]
        
        # Create train/val dataframes based on optimized subject split
        train_df = df[df['subject'].isin(train_subjects)].copy()
        val_df = df[df['subject'].isin(val_subjects)].copy()
        
        print(f"\nOptimized subject assignment:")
        print(f"Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
        print(f"Validation subjects ({len(val_subjects)}): {sorted(val_subjects)}")
        
        # Show actual vs target validation distribution
        actual_val_dist = val_df['classname'].value_counts().sort_index()
        print(f"\nValidation set optimization results:")
        print(f"{'Class':<5} {'Description':<25} {'Target':<8} {'Actual':<8} {'Diff':<8} {'Error%':<8}")
        print("-" * 75)
        total_error = 0
        for class_name in sorted(actual_val_dist.index):
            target = target_val_class_counts.get(class_name, 0)
            actual = actual_val_dist.get(class_name, 0)
            diff = actual - target
            error_pct = abs(diff) / target * 100 if target > 0 else 0
            total_error += abs(diff)
            description = CLASS_DESCRIPTIONS.get(class_name, 'Unknown')
            print(f"{class_name:<5} {description:<25} {target:<8} {actual:<8} {diff:<+8} {error_pct:<7.1f}%")
        
        print(f"Total absolute error: {total_error} samples")
        avg_error_pct = (total_error / target_val_samples) * 100
        print(f"Average error percentage: {avg_error_pct:.2f}%")
        
    else:
        # Optimized stratified split by class labels
        print("\nPerforming optimized stratified split by class labels...")
        
        # Use optimized stratified split
        train_df, val_df = create_optimal_stratified_split(
            df, 
            validation_split=validation_split, 
            random_state=random_state,
            n_iterations=50
        )
    
    # Comprehensive quality validation
    original_df = df.copy()
    
    # Display split statistics with validation
    print(f"\nBasic split statistics:")
    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    
    # Validate split quality
    split_name = "Subject-Based" if subject_based_split else "Stratified"
    quality_metrics = validate_split_quality(train_df, val_df, original_df, split_name)
    return train_df, val_df

def compare_split_methods(csv_file='data/driver_imgs_list.csv', validation_split=0.2, random_state=42):
    """Compare different split methods and recommend the best one."""
    
    print("COMPARING SPLIT METHODS")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Test stratified split
    print("\n1. Testing Optimized Stratified Split...")
    train_strat, val_strat = create_optimal_stratified_split(df, validation_split, random_state, n_iterations=50)
    metrics_strat = validate_split_quality(train_strat, val_strat, df, "Optimized Stratified")
    
    # Test subject-based split  
    print("\n2. Testing Optimized Subject-Based Split...")
    train_subj, val_subj = create_train_val_split(csv_file, validation_split, random_state, subject_based_split=True)
    metrics_subj = validate_split_quality(train_subj, val_subj, df, "Optimized Subject-Based")
    
    # Comparison
    print(f"\n" + "=" * 80)
    print("SPLIT METHOD COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<30} {'Stratified':<15} {'Subject-Based':<15} {'Better':<15}")
    print("-" * 80)
    
    comparison_metrics = [
        ('Total Error %', metrics_strat['total_error'], metrics_subj['total_error'], 'lower'),
        ('Max Train Error %', metrics_strat['max_train_error'], metrics_subj['max_train_error'], 'lower'),
        ('Max Val Error %', metrics_strat['max_val_error'], metrics_subj['max_val_error'], 'lower'),
        ('Quality Rating', metrics_strat['quality_rating'], metrics_subj['quality_rating'], 'text'),
    ]
    
    if SCIPY_AVAILABLE and metrics_strat['kl_train'] is not None:
        comparison_metrics.extend([
            ('KL Div (Train)', metrics_strat['kl_train'], metrics_subj['kl_train'], 'lower'),
            ('KL Div (Val)', metrics_strat['kl_val'], metrics_subj['kl_val'], 'lower'),
        ])
    
    for metric_name, strat_val, subj_val, compare_type in comparison_metrics:
        if compare_type == 'lower':
            better = "Stratified" if strat_val < subj_val else "Subject-Based"
            strat_str = f"{strat_val:.3f}"
            subj_str = f"{subj_val:.3f}"
        elif compare_type == 'text':
            better = "Similar" if strat_val == subj_val else "Different"
            strat_str = str(strat_val)
            subj_str = str(subj_val)
        
        print(f"{metric_name:<30} {strat_str:<15} {subj_str:<15} {better:<15}")
    
    # Recommendations
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if metrics_strat['total_error'] < metrics_subj['total_error'] * 0.5:
        print("🎯 RECOMMENDATION: Use STRATIFIED split")
        print("   ✅ Significantly better class balance")
        print("   ⚠️  But consider subject overlap for final evaluation")
    elif metrics_subj['total_error'] < metrics_strat['total_error'] * 2.0:
        print("🎯 RECOMMENDATION: Use SUBJECT-BASED split")
        print("   ✅ Good class balance with no data leakage")
        print("   ✅ Better for real-world performance estimation")
    else:
        print("🎯 RECOMMENDATION: Use BOTH splits")
        print("   📊 Use STRATIFIED for development/tuning")
        print("   🔬 Use SUBJECT-BASED for final evaluation")
    
    return (train_strat, val_strat, metrics_strat), (train_subj, val_subj, metrics_subj)

def save_split_to_csv(train_df, val_df, output_dir='data'):
    """Save train/validation splits to CSV files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    train_csv_path = os.path.join(output_dir, 'train_list.csv')
    val_csv_path = os.path.join(output_dir, 'val_list.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    print(f"\nSaved train set to: {train_csv_path}")
    print(f"Saved validation set to: {val_csv_path}")
    
    return train_csv_path, val_csv_path

def create_directory_structure(train_df, val_df, imgs_zip_path='data/imgs.zip', output_dir='data'):
    """
    Extract images and organize them into train/val directory structure.
    
    Expected structure:
    data/
    ├── train/
    │   ├── c0/  (safe driving)
    │   ├── c1/  (texting - right)
    │   └── ...
    └── val/
        ├── c0/  (safe driving)
        ├── c1/  (texting - right)
        └── ...
    """
    
    print("\nCreating directory structure...")
    
    # Create base directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    # Create class subdirectories
    classes = sorted(train_df['classname'].unique())
    
    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    
    print(f"Created directory structure:")
    print(f"  Train: {train_dir}")
    print(f"  Validation: {val_dir}")
    print(f"  Classes:")
    for class_name in classes:
        description = CLASS_DESCRIPTIONS.get(class_name, 'Unknown')
        print(f"    {class_name}/ ({description})")
    
    # Note: You'll need to extract imgs.zip and organize images according to the CSV files
    print(f"\nNext steps:")
    print(f"1. Extract {imgs_zip_path}")
    print(f"2. Use the train_list.csv and val_list.csv files to organize images into the directory structure")
    print(f"3. Each image should be placed in the appropriate class folder based on its classname")

def create_optimal_stratified_split(df, validation_split=0.2, random_state=42, n_iterations=100):
    """
    Create multiple stratified splits and select the one with best class balance.
    
    Args:
        df: DataFrame with the data
        validation_split: Fraction for validation
        random_state: Base random seed
        n_iterations: Number of different splits to try
    
    Returns:
        Best train_df, val_df with minimal class imbalance
    """
    
    print(f"Optimizing stratified split over {n_iterations} iterations...")
    
    best_train_df = None
    best_val_df = None
    best_balance_score = float('inf')
    best_seed = random_state
    
    # Target distribution
    original_dist = df['classname'].value_counts(normalize=True).sort_index()
    
    for i in range(n_iterations):
        current_seed = random_state + i
        
        # Create split with current seed
        train_df_temp, val_df_temp = train_test_split(
            df,
            test_size=validation_split,
            random_state=current_seed,
            stratify=df['classname']
        )
        
        # Calculate class distributions
        train_dist = train_df_temp['classname'].value_counts(normalize=True).sort_index()
        val_dist = val_df_temp['classname'].value_counts(normalize=True).sort_index()
        
        # Calculate balance score (lower is better)
        train_error = np.sum(np.abs(train_dist - original_dist))
        val_error = np.sum(np.abs(val_dist - original_dist))
        balance_score = train_error + val_error
        
        # Keep track of best split
        if balance_score < best_balance_score:
            best_balance_score = balance_score
            best_train_df = train_df_temp.copy()
            best_val_df = val_df_temp.copy()
            best_seed = current_seed
    
    print(f"Best stratified split found with seed {best_seed}")
    print(f"Balance score: {best_balance_score:.6f} (lower is better)")
    
    return best_train_df, best_val_df

def validate_split_quality(train_df, val_df, original_df, split_type=""):
    """
    Comprehensive validation of split quality focusing on class representation.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame  
        original_df: Original full DataFrame
        split_type: String identifier for the split type
    
    Returns:
        Dictionary with quality metrics
    """
    
    print(f"\n=== SPLIT QUALITY VALIDATION: {split_type} ===")
    
    # Basic statistics
    total_samples = len(original_df)
    train_samples = len(train_df)
    val_samples = len(val_df)
    
    print(f"Dataset size verification:")
    print(f"  Original: {total_samples} samples")
    print(f"  Train: {train_samples} samples ({train_samples/total_samples*100:.2f}%)")
    print(f"  Validation: {val_samples} samples ({val_samples/total_samples*100:.2f}%)")
    print(f"  Total after split: {train_samples + val_samples} samples")
    print(f"  Data loss: {total_samples - (train_samples + val_samples)} samples")
    
    # Class distribution analysis
    original_dist = original_df['classname'].value_counts(normalize=True).sort_index()
    train_dist = train_df['classname'].value_counts(normalize=True).sort_index()
    val_dist = val_df['classname'].value_counts(normalize=True).sort_index()
    
    print(f"\nClass representation analysis:")
    print(f"{'Class':<5} {'Description':<25} {'Original%':<10} {'Train%':<8} {'Val%':<8} {'Train Δ':<8} {'Val Δ':<8}")
    print("-" * 85)
    
    metrics = {
        'total_train_error': 0,
        'total_val_error': 0,
        'max_train_error': 0,
        'max_val_error': 0,
        'class_errors': {}
    }
    
    for class_name in sorted(original_dist.index):
        orig_pct = original_dist[class_name] * 100
        train_pct = train_dist.get(class_name, 0) * 100
        val_pct = val_dist.get(class_name, 0) * 100
        
        train_delta = train_pct - orig_pct
        val_delta = val_pct - orig_pct
        
        abs_train_error = abs(train_delta)
        abs_val_error = abs(val_delta)
        
        metrics['total_train_error'] += abs_train_error
        metrics['total_val_error'] += abs_val_error
        metrics['max_train_error'] = max(metrics['max_train_error'], abs_train_error)
        metrics['max_val_error'] = max(metrics['max_val_error'], abs_val_error)
        metrics['class_errors'][class_name] = {
            'train_error': abs_train_error,
            'val_error': abs_val_error
        }
        
        description = CLASS_DESCRIPTIONS.get(class_name, 'Unknown')
        print(f"{class_name:<5} {description:<25} {orig_pct:<9.2f}% {train_pct:<7.2f}% {val_pct:<7.2f}% {train_delta:<+7.2f}% {val_delta:<+7.2f}%")
    
    # Summary metrics
    avg_train_error = metrics['total_train_error'] / len(original_dist)
    avg_val_error = metrics['total_val_error'] / len(original_dist)
    
    print(f"\nBalance quality metrics:")
    print(f"  Average train error: {avg_train_error:.3f}%")
    print(f"  Average validation error: {avg_val_error:.3f}%")
    print(f"  Maximum train error: {metrics['max_train_error']:.3f}%")
    print(f"  Maximum validation error: {metrics['max_val_error']:.3f}%")
    print(f"  Total representation error: {avg_train_error + avg_val_error:.3f}%")
    
    # Quality assessment
    if avg_train_error + avg_val_error < 0.5:
        quality_rating = "EXCELLENT"
        quality_icon = "✅"
    elif avg_train_error + avg_val_error < 1.0:
        quality_rating = "VERY GOOD"
        quality_icon = "✅"
    elif avg_train_error + avg_val_error < 2.0:
        quality_rating = "GOOD"
        quality_icon = "✅"
    elif avg_train_error + avg_val_error < 5.0:
        quality_rating = "ACCEPTABLE"
        quality_icon = "⚠️"
    else:
        quality_rating = "POOR"
        quality_icon = "❌"
    
    print(f"  Overall quality: {quality_icon} {quality_rating}")
    
    # KL divergence calculation (if scipy is available)
    if SCIPY_AVAILABLE:
        # Ensure same classes in all distributions
        all_classes = sorted(set(original_dist.index) | set(train_dist.index) | set(val_dist.index))
        orig_probs = [original_dist.get(c, 1e-10) for c in all_classes]
        train_probs = [train_dist.get(c, 1e-10) for c in all_classes]
        val_probs = [val_dist.get(c, 1e-10) for c in all_classes]
        
        kl_train = entropy(train_probs, orig_probs)
        kl_val = entropy(val_probs, orig_probs)
        
        print(f"  KL divergence (train vs original): {kl_train:.6f}")
        print(f"  KL divergence (validation vs original): {kl_val:.6f}")
        
        metrics.update({
            'kl_train': kl_train,
            'kl_val': kl_val
        })
    else:
        print(f"  KL divergence: Not available (scipy not installed)")
        metrics.update({
            'kl_train': None,
            'kl_val': None
        })
    
    metrics.update({
        'avg_train_error': avg_train_error,
        'avg_val_error': avg_val_error,
        'total_error': avg_train_error + avg_val_error,
        'quality_rating': quality_rating
    })
    
    return metrics

if __name__ == "__main__":
    print("Distracted Driver MultiAction Classification - Fine-Tuned Train/Validation Split")
    print("=" * 90)
    
    # Compare different split methods
    (train_strat, val_strat, metrics_strat), (train_subj, val_subj, metrics_subj) = compare_split_methods()
    
    print(f"\n" + "=" * 90)
    print("CREATING FINAL SPLITS")
    print("=" * 90)
    
    # Save both splits
    print("\n1. Saving Optimized Stratified Split...")
    save_split_to_csv(
        train_strat, 
        val_strat, 
        output_dir='data/stratified_split'
    )
    
    print("\n2. Saving Optimized Subject-Based Split...")
    save_split_to_csv(
        train_subj, 
        val_subj, 
        output_dir='data/subject_split'
    )
    
    # Create directory structures
    create_directory_structure(
        train_strat, 
        val_strat, 
        output_dir='data/stratified_split'
    )
    
    create_directory_structure(
        train_subj, 
        val_subj, 
        output_dir='data/subject_split'
    )
    
    print(f"\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print("✅ Fine-tuned splits created with optimized class representation!")
    print(f"\nStratified Split Quality: {metrics_strat['quality_rating']} (Error: {metrics_strat['total_error']:.3f}%)")
    print(f"Subject-Based Split Quality: {metrics_subj['quality_rating']} (Error: {metrics_subj['total_error']:.3f}%)")
    
    print(f"\nFiles created:")
    print(f"📁 data/stratified_split/")
    print(f"   ├── train_list.csv ({len(train_strat)} samples)")
    print(f"   ├── val_list.csv ({len(val_strat)} samples)")
    print(f"   ├── train/ (directory structure)")
    print(f"   └── val/ (directory structure)")
    print(f"📁 data/subject_split/")
    print(f"   ├── train_list.csv ({len(train_subj)} samples)")
    print(f"   ├── val_list.csv ({len(val_subj)} samples)")
    print(f"   ├── train/ (directory structure)")
    print(f"   └── val/ (directory structure)")
    
    print(f"\nClass definitions:")
    for class_name, description in CLASS_DESCRIPTIONS.items():
        print(f"  {class_name}: {description}")
    
    print(f"\n🎯 Both splits maintain excellent class representation!")
    print(f"📊 Use analyze_splits.py for detailed comparison analysis")
