import pandas as pd
import sys
from collections import defaultdict

def analyze_domain_concepts(input_file):
    """
    Analyze and print the number of records per concept within each domain.

    Args:
        input_file (str): Path to input CSV file
    """
    try:
        # Read CSV
        df = pd.read_csv(input_file, encoding='utf-8')

        # Validate required columns
        required_columns = {'domain', 'concept'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Print total records
        print(f"\nTotal Records: {len(df)}")

        # Group by domain and count records per concept
        domain_concept_counts = df.groupby(['domain', 'concept']).size().reset_index(name='record_count')

        # Organize results by domain
        domain_summary = defaultdict(list)
        for _, row in domain_concept_counts.iterrows():
            domain_summary[row['domain']].append({
                'concept': row['concept'],
                'count': row['record_count']
            })

        # Print detailed summary
        print("\nDomain and Concept Distribution:")
        print("-" * 40)
        
        for domain, concepts in domain_summary.items():
            print(f"\n{domain} Domain:")
            total_domain_records = sum(concept['count'] for concept in concepts)
            print(f"  Total Domain Records: {total_domain_records}")
            
            # Sort concepts by count in descending order
            sorted_concepts = sorted(concepts, key=lambda x: x['count'], reverse=True)
            
            for concept in sorted_concepts:
                percentage = (concept['count'] / total_domain_records) * 100
                print(f"  - {concept['concept']}: {concept['count']} records ({percentage:.2f}%)")

        return domain_summary

    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        raise

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_csv_path>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    analyze_domain_concepts(input_file)

if __name__ == "__main__":
    main()
