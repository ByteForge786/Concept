import pandas as pd
import numpy as np
from pathlib import Path
import random

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
        ndm_df = pd.read_csv(ndm_path, encoding='utf-8')
        master_df = pd.read_csv(master_path, encoding='utf-8')
        
        # Validate required columns
        if not {'attribute_name', 'domain', 'concept'}.issubset(ndm_df.columns):
            raise ValueError("NDM CSV must contain attribute_name, domain, and concept columns")
        if not {'attribute_name', 'description'}.issubset(master_df.columns):
            raise ValueError("Master CSV must contain attribute_name and description columns")
        
        # Merge dataframes on attribute_name
        merged_df = pd.merge(
            ndm_df[['attribute_name', 'domain', 'concept']],
            master_df[['attribute_name', 'description']],
            on='attribute_name',
            how='inner'
        )
        
        # Remove any rows with missing values
        merged_df = merged_df.dropna()
        
        # Reorder columns
        merged_df = merged_df[['attribute_name', 'description', 'domain', 'concept']]
        
        # Create input.csv with all complete records
        input_path = output_path / 'input.csv'
        merged_df.to_csv(input_path, index=False, encoding='utf-8')
        
        # Create test.csv with some blank domains/concepts
        test_df = merged_df.copy()
        num_rows = len(test_df)
        
        # Randomly select rows to blank out domain, concept, or both
        blank_mask = np.random.choice([1, 2, 3], size=num_rows, p=[0.3, 0.3, 0.4])
        
        for idx, mask in enumerate(blank_mask):
            if mask == 1:  # Blank domain
                test_df.loc[idx, 'domain'] = ''
            elif mask == 2:  # Blank concept
                test_df.loc[idx, 'concept'] = ''
            elif mask == 3:  # Blank both
                test_df.loc[idx, ['domain', 'concept']] = ''
        
        # Save test.csv
        test_path = output_path / 'test.csv'
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        
        # Print summary
        print(f"\nProcessing completed successfully!")
        print(f"Total records processed: {len(merged_df)}")
        print(f"Files created:")
        print(f"1. Input file: {input_path}")
        print(f"   - Contains {len(merged_df)} complete records")
        print(f"2. Test file: {test_path}")
        print(f"   - Contains {len(test_df)} records with some blank values")
        
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
        print("\nYou can now use these files for training and testing:")
        print(f"Training: python classifier.py --mode train --input_file {input_path}")
        print(f"Testing: python classifier.py --mode predict --input_file {test_path}")
    except Exception as e:
        print(f"Failed to process files: {str(e)}")
