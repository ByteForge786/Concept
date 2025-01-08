import pandas as pd
import sys
from collections import defaultdict

def clean_and_analyze_csv(input_file, output_file=None):
    """
    Clean CSV by removing duplicates while keeping the first instance
    and provide detailed statistics.

    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to save cleaned CSV. 
                                     If None, will use input filename with '_cleaned' suffix
    """
    try:
        # Read CSV
        df = pd.read_csv(input_file, encoding='utf-8')

        # Validate required columns
        required_columns = {'attribute_name', 'domain', 'concept'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Print initial statistics
        print("\n--- Initial Dataset Statistics ---")
        print(f"Total records: {len(df)}")
        
        # Count initial records per domain
        initial_domain_counts = df['domain'].value_counts()
        print("\nInitial Records per Domain:")
        for domain, count in initial_domain_counts.items():
            print(f"{domain}: {count} records")

        # Identify duplicates
        duplicate_mask = df.duplicated(subset=['attribute_name', 'domain', 'concept'], keep='first')
        total_duplicate_records = duplicate_mask.sum()
        
        print(f"\nTotal duplicate records: {total_duplicate_records}")

        # Create cleaned DataFrame by dropping duplicates
        # keep='first' ensures the first occurrence of a duplicate set is kept
        cleaned_df = df.drop_duplicates(subset=['attribute_name', 'domain', 'concept'], keep='first')

        # Print cleaning statistics
        print(f"Total records after cleaning: {len(cleaned_df)}")
        print(f"Records removed: {len(df) - len(cleaned_df)}")

        # Final domain statistics
        print("\n--- Final Dataset Statistics ---")
        final_domain_counts = cleaned_df['domain'].value_counts()
        for domain, count in final_domain_counts.items():
            print(f"{domain}: {count} records")
        
        # Detailed concept statistics per domain
        print("\n--- Concept Distribution per Domain ---")
        concept_distribution = defaultdict(lambda: defaultdict(int))
        for _, row in cleaned_df.iterrows():
            concept_distribution[row['domain']][row['concept']] += 1
        
        for domain, concepts in concept_distribution.items():
            print(f"\n{domain} Domain:")
            for concept, count in concepts.items():
                print(f"  - {concept}: {count} records")

        # Save cleaned CSV if output file specified
        if output_file is None:
            output_file = input_file.replace('.csv', '_cleaned.csv')
        
        cleaned_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nCleaned CSV saved to: {output_file}")

        return cleaned_df

    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        raise

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_csv_path> [output_csv_path]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    clean_and_analyze_csv(input_file, output_file)

if __name__ == "__main__":
    main()
