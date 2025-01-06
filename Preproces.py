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
