import pandas as pd
import sys
from collections import defaultdict

def advanced_deduplicate_and_analyze(input_file, output_file=None):
    """
    Advanced CSV deduplication with detailed statistics.
    
    Duplicate Removal Logic:
    1. If attribute_name, domain, concept are EXACTLY the same - keep first instance
    2. If attribute_name is same but domain or concept differ - keep all instances
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to save cleaned CSV
    """
    try:
        # Read CSV
        df = pd.read_csv(input_file, encoding='utf-8')

        # Validate required columns
        required_columns = {'attribute_name', 'domain', 'concept'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # BEFORE Deduplication Analysis
        print("\n--- BEFORE Deduplication ---")
        print(f"Total Initial Records: {len(df)}")
        
        # Identify different types of duplicates
        # 1. Exact duplicates (all columns same)
        exact_duplicates = df.duplicated(subset=['attribute_name', 'domain', 'concept'], keep=False)
        exact_duplicate_count = exact_duplicates.sum()
        
        # 2. Partial duplicates (same attribute_name, different domain/concept)
        # First, find all attribute names with multiple entries
        multi_attribute_names = df[df.duplicated(subset=['attribute_name'], keep=False)]['attribute_name'].unique()
        
        # Track partial duplicates
        partial_duplicates = []
        for attr_name in multi_attribute_names:
            attr_group = df[df['attribute_name'] == attr_name]
            # If group has multiple unique domain/concept combinations
            if len(attr_group[['domain', 'concept']].drop_duplicates()) > 1:
                partial_duplicates.extend(attr_group.index.tolist())
        
        partial_duplicate_count = len(set(partial_duplicates))
        
        # Initial domain and concept analysis
        print("\nInitial Domain Distribution:")
        initial_domain_counts = df['domain'].value_counts()
        for domain, count in initial_domain_counts.items():
            print(f"{domain}: {count} records")
        
        # Perform advanced deduplication
        # Step 1: Remove exact duplicates, keeping first instance
        df_cleaned = df.drop_duplicates(subset=['attribute_name', 'domain', 'concept'], keep='first')
        
        # Step 2: Preserve entries with same attribute_name but different domain/concept
        # We do this by using the multi_attribute_names we identified earlier
        
        # AFTER Deduplication Analysis
        print("\n--- AFTER Deduplication ---")
        print(f"Total Records After Cleaning: {len(df_cleaned)}")
        print(f"Exact Duplicates Removed: {exact_duplicate_count}")
        print(f"Attributes with Multiple Variations: {partial_duplicate_count}")
        
        # Final domain distribution
        print("\nFinal Domain Distribution:")
        final_domain_counts = df_cleaned['domain'].value_counts()
        for domain, count in final_domain_counts.items():
            print(f"{domain}: {count} records")
        
        # Detailed concept statistics per domain
        print("\n--- Concept Distribution per Domain ---")
        concept_distribution = defaultdict(lambda: defaultdict(int))
        for _, row in df_cleaned.iterrows():
            concept_distribution[row['domain']][row['concept']] += 1
        
        for domain, concepts in concept_distribution.items():
            print(f"\n{domain} Domain:")
            total_domain_records = sum(concepts.values())
            
            # Sort concepts by count in descending order
            sorted_concepts = sorted(concepts.items(), key=lambda x: x[1], reverse=True)
            
            for concept, count in sorted_concepts:
                percentage = (count / total_domain_records) * 100
                print(f"  - {concept}: {count} records ({percentage:.2f}%)")
        
        # Save cleaned CSV
        if output_file is None:
            output_file = input_file.replace('.csv', '_cleaned.csv')
        
        df_cleaned.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nCleaned CSV saved to: {output_file}")
        
        return df_cleaned

    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        raise

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_csv_path> [output_csv_path]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    advanced_deduplicate_and_analyze(input_file, output_file)

if __name__ == "__main__":
    main()
