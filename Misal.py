class SampleGenerator:
    """Handles synthetic sample generation with parallel processing."""
    
    def __init__(self, batch_size: int = BATCH_SIZE, definitions_path: str = 'definitions.txt'):
        self.batch_size = batch_size
        self.used_samples_lock = Lock()
        self.retry_handler = AdaptiveRetryHandler()
        self.generation_stats = {
            'attempts': 0,
            'successes': 0,
            'failures': 0
        }
        self.definitions = self._load_definitions(definitions_path)
        
    def _load_definitions(self, definitions_path: str) -> Dict[Tuple[str, str], str]:
        """Load domain-concept definitions from file."""
        definitions = {}
        try:
            with open(definitions_path, 'r') as f:
                for line in f:
                    if line.strip():
                        domain, concept, definition = line.strip().split('|')
                        definitions[(domain.strip(), concept.strip())] = definition.strip()
            return definitions
        except Exception as e:
            logger.warning(f"Failed to load definitions: {str(e)}")
            return {}

    def generate_prompt(self, 
                       domain: str,
                       concept: str,
                       context_samples: List[Dict],
                       batch_size: int) -> str:
        context_str = "\n".join([
            f"Example {i+1}:\nAttribute: {s['attribute_name']}\nDescription: {s['description']}"
            for i, s in enumerate(context_samples)
        ])
        
        # Get definition for this domain-concept pair
        definition = self.definitions.get((domain, concept), "")
        definition_str = f"\nDefinition of {domain} - {concept}:\n{definition}\n" if definition else ""
        
        return f"""Based on these examples from {domain} - {concept}:{definition_str}
{context_str}

Generate {batch_size} new, unique analytics attributes.
Each must follow the pattern shown in examples but be distinctly different.

Requirements:
1. Attribute names must be in snake_case
2. Descriptions should be clear and concise
3. Must be different from examples
4. Follow the exact pattern seen in examples
5. Attributes should align with the provided domain-concept definition

Provide exactly {batch_size} responses in this format, one per line:
{{"attribute_name": "name", "description": "description"}}

Output {batch_size} lines of JSON only, no additional text."""





class DataBalancer:
    """Main class for data balancing operations."""
    
    def __init__(self,
                 max_workers: int = MAX_WORKERS,
                 batch_size: int = BATCH_SIZE,
                 definitions_path: str = 'definitions.txt'):
        self.max_workers = max_workers
        self.sample_generator = SampleGenerator(batch_size, definitions_path)
        self.embedding_manager = ParallelEmbeddingManager(max_workers=max_workers)





def main(input_path: str,
         output_dir: str = 'data',
         max_workers: int = MAX_WORKERS,
         batch_size: int = BATCH_SIZE,
         definitions_path: str = 'definitions.txt'):
    """Main execution function."""
    try:
        # ... (previous code remains the same until balancer initialization)
        
        # Initialize balancer with definitions path
        balancer = DataBalancer(max_workers, batch_size, definitions_path)
        
        # ... (rest of the code remains the same)

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
    parser.add_argument('--definitions',
                       default='definitions.txt',
                       help='Path to domain-concept definitions file')
    
    args = parser.parse_args()
    
    try:
        main(
            args.input,
            args.output_dir,
            args.max_workers,
            args.batch_size,
            args.definitions
        )
        logger.info("Processing completed successfully!")
    except Exception as e:
        logger.error(f"Failed to process dataset: {str(e)}")
        raise
