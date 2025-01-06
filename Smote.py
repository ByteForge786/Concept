import pandas as pd
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
import numpy as np

# Example original data
original_data = {
    'attribute_name': [
        'monthly_revenue',
        'daily_revenue',
        'active_users',
        'dau',
        'mau'
    ],
    'description': [
        'Monthly revenue from all product sales',
        'Daily revenue from subscriptions',
        'Number of users who performed any action',
        'Daily active users in the platform',
        'Monthly active users count'
    ],
    'domain': [
        'Financial',
        'Financial',
        'User Analytics',
        'User Analytics',
        'User Analytics'
    ],
    'concept': [
        'Revenue',
        'Revenue',
        'Usage',
        'Usage',
        'Usage'
    ]
}

def show_smote_example():
    # Create DataFrame
    df = pd.DataFrame(original_data)
    
    print("Original Data Distribution:")
    print("=" * 50)
    print(df.groupby(['domain', 'concept']).size())
    print("\nOriginal Samples:")
    print("-" * 50)
    for _, row in df.iterrows():
        print(f"{row['domain']} - {row['concept']}: {row['description']}")
    
    # Convert text to embeddings
    encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = encoder.encode(df['description'].tolist())
    
    # Create combined target
    df['target'] = df['domain'] + '_' + df['concept']
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(embeddings, df['target'])
    
    # Generate new descriptions by finding closest original descriptions
    new_samples = []
    for idx, embedding in enumerate(X_resampled):
        # If it's a new synthetic sample
        if idx >= len(df):
            # Find most similar original description
            similarities = embeddings @ embedding.T
            closest_idx = np.argmax(similarities)
            domain, concept = y_resampled[idx].split('_')
            
            # Create synthetic description by modifying closest match
            original_desc = df.iloc[closest_idx]['description']
            synthetic_desc = f"SYNTHETIC - Based on: {original_desc}"
            
            new_samples.append({
                'attribute_name': f'synthetic_{idx}',
                'description': synthetic_desc,
                'domain': domain,
                'concept': concept
            })
    
    # Create balanced DataFrame
    balanced_df = pd.concat([
        df,
        pd.DataFrame(new_samples)
    ], ignore_index=True)
    
    print("\nBalanced Data Distribution:")
    print("=" * 50)
    print(balanced_df.groupby(['domain', 'concept']).size())
    print("\nNew Synthetic Samples:")
    print("-" * 50)
    for _, row in balanced_df[len(df):].iterrows():
        print(f"{row['domain']} - {row['concept']}: {row['description']}")

if __name__ == "__main__":
    show_smote_example()
