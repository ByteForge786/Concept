import pandas as pd
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict

# More comprehensive example data with edge cases
original_data = {
    'attribute_name': [
        # Financial metrics (many samples)
        'monthly_revenue', 'daily_revenue', 'quarterly_revenue', 'annual_revenue', 'subscription_revenue',
        # User Analytics (medium samples)
        'dau', 'mau', 'wau',
        # Performance (few samples)
        'latency',
        # Security (single sample)
        'failed_logins'
    ],
    'description': [
        # Financial descriptions
        'Monthly revenue from all product sales',
        'Daily revenue from subscriptions',
        'Quarterly revenue including all sources',
        'Annual revenue for fiscal year',
        'Monthly recurring revenue from subscriptions',
        # User Analytics descriptions
        'Daily active users in the platform',
        'Monthly active users count',
        'Weekly active users in system',
        # Performance description
        'Average response time in milliseconds',
        # Security description
        'Number of failed login attempts'
    ],
    'domain': [
        # Domains
        'Financial', 'Financial', 'Financial', 'Financial', 'Financial',
        'User Analytics', 'User Analytics', 'User Analytics',
        'Performance', 
        'Security'
    ],
    'concept': [
        # Concepts
        'Revenue', 'Revenue', 'Revenue', 'Revenue', 'Revenue',
        'Usage', 'Usage', 'Usage',
        'Latency',
        'Access'
    ]
}

def dynamic_smote_balance(df: pd.DataFrame, target_samples: int = None):
    """
    Balance dataset using SMOTE with dynamic neighbors per class.
    
    Args:
        df: Input DataFrame
        target_samples: Target number of samples per class (default: size of largest class)
    """
    print("\nOriginal Data Distribution:")
    print("=" * 50)
    class_counts = df.groupby(['domain', 'concept']).size()
    print(class_counts)
    
    # If target_samples not specified, use size of largest class
    if target_samples is None:
        target_samples = class_counts.max()
    
    # Convert text to embeddings
    encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = encoder.encode(df['description'].tolist())
    
    # Create combined target
    df['target'] = df['domain'] + '_' + df['concept']
    target_counts = df['target'].value_counts()
    
    # Process each class separately with dynamic k_neighbors
    balanced_samples = []
    
    for target in target_counts.index:
        domain, concept = target.split('_')
        class_mask = (df['target'] == target)
        class_embeddings = embeddings[class_mask]
        class_df = df[class_mask]
        
        n_samples = len(class_df)
        
        print(f"\nProcessing {domain} - {concept}")
        print(f"Original samples: {n_samples}")
        
        if n_samples == 1:
            print("Single sample case - Using data augmentation")
            # For single sample, create variations
            original_desc = class_df['description'].iloc[0]
            for i in range(target_samples - 1):
                # Create variations by adding qualifiers
                variations = [
                    f"Alternative {i+1}: {original_desc}",
                    f"Variant {i+1} of: {original_desc}",
                    f"Similar to: {original_desc}"
                ]
                synthetic_desc = np.random.choice(variations)
                balanced_samples.append({
                    'attribute_name': f'synthetic_{domain}_{concept}_{i}',
                    'description': synthetic_desc,
                    'domain': domain,
                    'concept': concept,
                    'target': target
                })
        elif n_samples < target_samples:
            print(f"Applying SMOTE with k_neighbors={n_samples-1}")
            # Use all available samples as neighbors
            smote = SMOTE(
                random_state=42,
                k_neighbors=n_samples-1,
                n_jobs=-1
            )
            
            # Calculate how many synthetic samples needed
            n_synthetic = target_samples - n_samples
            
            try:
                X_resampled, y_resampled = smote.fit_resample(
                    class_embeddings,
                    [1] * len(class_embeddings)
                )
                
                # Generate synthetic descriptions
                for i in range(n_samples, len(X_resampled)):
                    # Find closest original description
                    similarities = class_embeddings @ X_resampled[i].T
                    closest_idx = np.argmax(similarities)
                    original_desc = class_df['description'].iloc[closest_idx]
                    
                    synthetic_desc = f"SMOTE Generated ({i-n_samples+1}): Based on {original_desc}"
                    
                    balanced_samples.append({
                        'attribute_name': f'synthetic_{domain}_{concept}_{i}',
                        'description': synthetic_desc,
                        'domain': domain,
                        'concept': concept,
                        'target': target
                    })
            except ValueError as e:
                print(f"SMOTE failed: {e}. Using simple replication.")
                # Fallback to simple replication
                for i in range(n_synthetic):
                    original_idx = i % n_samples
                    original_desc = class_df['description'].iloc[original_idx]
                    synthetic_desc = f"Replicated ({i+1}): {original_desc}"
                    
                    balanced_samples.append({
                        'attribute_name': f'synthetic_{domain}_{concept}_{i}',
                        'description': synthetic_desc,
                        'domain': domain,
                        'concept': concept,
                        'target': target
                    })
        
        # Keep original samples
        balanced_samples.extend(class_df.to_dict('records'))
    
    # Create final balanced DataFrame
    balanced_df = pd.DataFrame(balanced_samples)
    
    print("\nFinal Balanced Distribution:")
    print("=" * 50)
    print(balanced_df.groupby(['domain', 'concept']).size())
    
    # Show examples of synthetic samples
    print("\nSynthetic Sample Examples:")
    print("-" * 50)
    synthetic_samples = balanced_df[balanced_df['attribute_name'].str.startswith('synthetic_')]
    for domain in synthetic_samples['domain'].unique():
        domain_samples = synthetic_samples[synthetic_samples['domain'] == domain]
        print(f"\n{domain}:")
        for _, row in domain_samples.head(2).iterrows():
            print(f"{row['concept']}: {row['description']}")
    
    return balanced_df

# Run example
if __name__ == "__main__":
    df = pd.DataFrame(original_data)
    balanced_df = dynamic_smote_balance(df)










import pandas as pd
import numpy as np
from pathlib import Path
import random
from collections import defaultdict

def analyze_distribution(df: pd.DataFrame) -> str:
    """
    Analyze and format the distribution of domains and concepts.
    """
    # Initialize string buffer for output
    output = "\nData Distribution Analysis:\n" + "="*50 + "\n"
    
    # Get domain-concept hierarchy
    hierarchy = defaultdict(list)
    for domain, concept_group in df.groupby('domain')['concept']:
        concepts = concept_group.value_counts()
        hierarchy[domain] = concepts
    
    # Calculate and format statistics
    total_records = len(df)
    output += f"Total Records: {total_records}\n\n"
    
    # Domain level statistics
    domain_counts = df['domain'].value_counts()
    output += "Domain Distribution:\n" + "-"*30 + "\n"
    for domain, count in domain_counts.items():
        percentage = (count / total_records) * 100
        output += f"{domain}: {count} records ({percentage:.2f}%)\n"
        
        # Concept level statistics for this domain
        domain_concepts = hierarchy[domain]
        output += "\tConcepts within this domain:\n"
        for concept, concept_count in domain_concepts.items():
            concept_percentage = (concept_count / count) * 100
            output += f"\t- {concept}: {concept_count} records ({concept_percentage:.2f}% of domain)\n"
        output += "\n"
    
    return output

def process_csvs(ndm_path: str, master_path: str, output_dir: str = 'data'):
    """
    Process NDM and Master Definition CSVs to create training and test datasets.
    
    Args:
        ndm_path: Path to NDM CSV file containing attribute_name, domain, concept
        master_path: Path to Master Definition CSV containing attribute_name, description
        output_dir: Directory to save output files (default: 'data')
    """
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read CSVs with UTF-8 encoding
        print("\nReading input files...")
        ndm_df = pd.read_csv(ndm_path, encoding='utf-8')
        master_df = pd.read_csv(master_path, encoding='utf-8')
        
        # Validate required columns
        required_ndm = {'attribute_name', 'domain', 'concept'}
        required_master = {'attribute_name', 'description'}
        
        missing_ndm = required_ndm - set(ndm_df.columns)
        missing_master = required_master - set(master_df.columns)
        
        if missing_ndm:
            raise ValueError(f"NDM CSV missing columns: {missing_ndm}")
        if missing_master:
            raise ValueError(f"Master CSV missing columns: {missing_master}")
        
        # Merge dataframes on attribute_name
        print("Merging datasets...")
        merged_df = pd.merge(
            ndm_df[['attribute_name', 'domain', 'concept']],
            master_df[['attribute_name', 'description']],
            on='attribute_name',
            how='inner'
        )
        
        # Remove any rows with missing values
        initial_len = len(merged_df)
        merged_df = merged_df.dropna()
        dropped_count = initial_len - len(merged_df)
        if dropped_count > 0:
            print(f"Dropped {dropped_count} rows with missing values")
        
        # Reorder columns
        merged_df = merged_df[['attribute_name', 'description', 'domain', 'concept']]
        
        # Analyze and print distribution
        distribution_analysis = analyze_distribution(merged_df)
        print(distribution_analysis)
        
        # Save distribution analysis to file
        with open(output_path / 'distribution_analysis.txt', 'w', encoding='utf-8') as f:
            f.write(distribution_analysis)
        
        # Create input.csv with all complete records
        input_path = output_path / 'input.csv'
        merged_df.to_csv(input_path, index=False, encoding='utf-8')
        
        # Create test.csv with some blank domains/concepts
        test_df = merged_df.copy()
        num_rows = len(test_df)
        
        # Randomly select rows to blank out domain, concept, or both
        blank_mask = np.random.choice([1, 2, 3], size=num_rows, p=[0.3, 0.3, 0.4])
        
        blanked_domains = 0
        blanked_concepts = 0
        blanked_both = 0
        
        for idx, mask in enumerate(blank_mask):
            if mask == 1:  # Blank domain
                test_df.loc[idx, 'domain'] = ''
                blanked_domains += 1
            elif mask == 2:  # Blank concept
                test_df.loc[idx, 'concept'] = ''
                blanked_concepts += 1
            elif mask == 3:  # Blank both
                test_df.loc[idx, ['domain', 'concept']] = ''
                blanked_both += 1
        
        # Save test.csv
        test_path = output_path / 'test.csv'
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        
        # Print summary
        print(f"\nProcessing Summary:")
        print(f"=" * 50)
        print(f"Total records processed: {len(merged_df)}")
        print(f"\nFiles created:")
        print(f"1. Input file: {input_path}")
        print(f"   - Contains {len(merged_df)} complete records")
        print(f"2. Test file: {test_path}")
        print(f"   - Contains {len(test_df)} records")
        print(f"   - Blanked domains only: {blanked_domains}")
        print(f"   - Blanked concepts only: {blanked_concepts}")
        print(f"   - Blanked both: {blanked_both}")
        print(f"3. Distribution Analysis: {output_path}/distribution_analysis.txt")
        
        # Return paths for further use
        return str(input_path), str(test_path)
        
    except Exception as e:
        print(f"Error processing CSVs: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    ndm_path = "ndm_correct.csv"
    master_path = "master_definition.csv"
    
    try:
        input_path, test_path = process_csvs(ndm_path, master_path)
        print("\nNext Steps:")
        print(f"1. Train the model:")
        print(f"   python classifier.py --mode train --input_file {input_path}")
        print(f"\n2. Test the model:")
        print(f"   python classifier.py --mode predict --input_file {test_path}")
    except Exception as e:
        print(f"Failed to process files: {str(e)}")
