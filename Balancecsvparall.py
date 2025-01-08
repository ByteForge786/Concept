for sample in samples:
                used_samples.add(f"{sample['attribute_name']}_{sample['description']}")
        
        # Create final dataframes
        training_df = pd.DataFrame(training_samples)
        eval_df = pd.DataFrame(eval_samples)
        
        # Log generation statistics
        self.sample_generator.log_generation_stats()
        
        return training_df, eval_df

def main(input_path: str,
         output_dir: str = 'data',
         max_workers: int = MAX_WORKERS,
         batch_size: int = BATCH_SIZE):
    """Main execution function."""
    try:
        # Setup directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging file
        log_path = output_path / 'balancer.log'
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Read input data
        logger.info(f"Reading data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Validate input data
        required_columns = {'attribute_name', 'description', 'domain', 'concept'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Input CSV must contain columns: {required_columns}")
        
        # Log initial statistics
        logger.info("\nInitial Data Statistics:")
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Unique domains: {df['domain'].nunique()}")
        logger.info(f"Unique concepts: {df['concept'].nunique()}")
        logger.info("\nDomain-Concept Distribution:")
        logger.info(df.groupby(['domain', 'concept']).size())
        
        # Initialize balancer
        balancer = DataBalancer(max_workers, batch_size)
        
        # Process data
        start_time = time.time()
        training_df, eval_df = balancer.balance_dataset(df)
        processing_time = time.time() - start_time
        
        # Save results
        training_path = output_path / 'balanced_training_data.csv'
        eval_path = output_path / 'evaluation_data.csv'
        original_path = output_path / 'original_data.csv'
        
        df.to_csv(original_path, index=False)
        training_df.to_csv(training_path, index=False)
        eval_df.to_csv(eval_path, index=False)
        
        # Generate detailed report
        report = [
            "Data Balance Report",
            "=" * 50,
            f"\nProcessing Configuration:",
            f"Max Workers: {max_workers}",
            f"Batch Size: {batch_size}",
            f"Processing Time: {processing_time:.2f} seconds",
            f"\nOriginal Data Statistics:",
            f"Total Records: {len(df)}",
            f"Domains: {df['domain'].nunique()}",
            f"Concepts: {df['concept'].nunique()}",
            "\nOriginal Distribution:",
            str(df.groupby(['domain', 'concept']).size()),
            "\nBalanced Training Set Distribution:",
            str(training_df.groupby(['domain', 'concept']).size()),
            "\nEvaluation Set Distribution:",
            str(eval_df.groupby(['domain', 'concept']).size()),
            f"\nProcessing Performance:",
            f"Average time per sample: {processing_time/len(training_df):.3f} seconds",
            f"Samples per second: {len(training_df)/processing_time:.2f}",
            f"\nSynthetic Sample Statistics:",
            f"Original samples: {len(df)}",
            f"Training samples: {len(training_df)}",
            f"Evaluation samples: {len(eval_df)}",
            f"Generated samples: {len(training_df) - (len(df) - len(eval_df))}",
            f"\nRate Limiting Statistics:",
            f"Average delay between requests: {balancer.sample_generator.retry_handler.rate_limiter.current_delay:.2f}s",
            f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}",
            f"Final error rate: {sum(balancer.sample_generator.retry_handler.rate_limiter.error_window)/len(balancer.sample_generator.retry_handler.rate_limiter.error_window) if balancer.sample_generator.retry_handler.rate_limiter.error_window else 0:.2%}"
        ]
        
        # Save report
        report_path = output_path / 'balance_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"\nProcessing complete!")
        logger.info(f"Files saved:")
        logger.info(f"1. Original data: {original_path}")
        logger.info(f"2. Training data: {training_path}")
        logger.info(f"3. Evaluation data: {eval_path}")
        logger.info(f"4. Balance report: {report_path}")
        logger.info(f"5. Process log: {log_path}")
        
        return str(training_path), str(eval_path)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Balance dataset with adaptive rate limiting')
    parser.add_argument('--input', 
                       required=True,
                       help='Input CSV file path')
    parser.add_argument('--output-dir',
                       default='data',
                       help='Output directory')
    parser.add_argument('--max-workers',
                       type=int,
                       default=MAX_WORKERS,
                       help='Maximum number of parallel workers')
    parser.add_argument('--batch-size',
                       type=int,
                       default=BATCH_SIZE,
                       help='Batch size for generation')
    
    args = parser.parse_args()
    
    try:
        main(
            args.input,
            args.output_dir,
            args.max_workers,
            args.batch_size
        )
        logger.info("Processing completed successfully!")
    except Exception as e:
        logger.error(f"Failed to process dataset: {str(e)}")
        raise
