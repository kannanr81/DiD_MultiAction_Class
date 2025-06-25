import pandas as pd
import numpy as np

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

# Load the data
df = pd.read_csv('data/driver_imgs_list.csv')

# Analyze class distribution
print("Distracted Driver MultiAction Classification Dataset")
print("=" * 60)
print("Class distribution:")
class_counts = df['classname'].value_counts().sort_index()
print(class_counts)

print(f"\nTotal samples: {len(df)}")
print(f"Number of unique subjects: {df['subject'].nunique()}")

# Show class percentages with descriptions
print("\nDetailed class analysis:")
print(f"{'Class':<5} {'Description':<25} {'Count':<8} {'Percentage':<10}")
print("-" * 55)
for class_name in sorted(df['classname'].unique()):
    count = len(df[df['classname'] == class_name])
    percentage = (count / len(df)) * 100
    description = CLASS_DESCRIPTIONS.get(class_name, 'Unknown')
    print(f"{class_name:<5} {description:<25} {count:<8} {percentage:<9.2f}%")

# Show unique subjects
print(f"\nUnique subjects: {sorted(df['subject'].unique())}")

# Analyze subject-class distribution
print(f"\nSubject-class distribution analysis:")
subject_class_counts = df.groupby(['subject', 'classname']).size().unstack(fill_value=0)
print(f"Subjects per class:")
for class_name in sorted(df['classname'].unique()):
    subjects_with_class = (subject_class_counts[class_name] > 0).sum()
    description = CLASS_DESCRIPTIONS.get(class_name, 'Unknown')
    print(f"  {class_name} ({description}): {subjects_with_class} subjects")

# Check for potential data imbalance
print(f"\nDataset balance analysis:")
min_class_count = class_counts.min()
max_class_count = class_counts.max()
imbalance_ratio = max_class_count / min_class_count
print(f"  Most represented class: {class_counts.idxmax()} with {max_class_count} samples")
print(f"  Least represented class: {class_counts.idxmin()} with {min_class_count} samples")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}")

if imbalance_ratio > 2.0:
    print("  ⚠️  Dataset shows significant class imbalance")
else:
    print("  ✅ Dataset is reasonably balanced")
