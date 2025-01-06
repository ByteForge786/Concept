import pandas as pd
import numpy as np
from pathlib import Path
import faiss
import pickle
from typing import List, Dict, Tuple, Set, Optional, Iterator
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from tqdm import tqdm
from test import *  # for chinou_response
import json
import logging
import time
from threading import Lock, local
from queue import Queue
import os

# Constants
SAMPLES_PER_CLASS = 500
BATCH_SIZE = 5
MAX_RETRIES = 3
MAX_WORKERS = 4
CONTEXT_SIZE = 6

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GenerationTask:
    domain: str
    concept: str
    context_samples: List[Dict]
    num_needed: int
    used_samples: Set[str]

class ThreadLocalLLM:
    """Thread-local LLM handler with independent rate limiting."""
    
    def __init__(self, initial_delay: float = 0.1):
        self.delay = initial_delay
        self.last_call_time = 0
        self.success_streak = 0
        self.failure_streak = 0
        self.lock = Lock()

    def wait_if_needed(self):
        """Thread-safe wait before API call."""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_call_time
            if time_since_last < self.delay:
                time.sleep(self.delay - time_since_last)
            self.last_call_time = time.time()

    def adjust_delay(self, success: bool):
        """Adjust delay based on success/failure."""
        with self.lock:
            if success:
                self.success_streak += 1
                self.failure_streak = 0
                if self.success_streak >= 5:
                    self.delay = max(0.1, self.delay * 0.8)
                    self.success_streak = 0
            else:
                self.failure_streak += 1
                self.success_streak = 0
                self.delay = min(5.0, self.delay * 2)

class ParallelSampleGenerator:
    """Generates samples using parallel LLM calls."""
    
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.thread_local = local()
        self.used_samples_lock = Lock()
        self.stats_lock = Lock()
        self.generation_stats = {
            'attempts': 0,
            'successes': 0,
            'failures': 0
        }

    def get_llm_handler(self) -> ThreadLocalLLM:
        """Get or create thread-local LLM handler."""
        if not hasattr(self.thread_local, 'llm_handler'):
            self.thread_local.llm_handler = ThreadLocalLLM()
        return self.thread_local.llm_handler

    def generate_prompt(self, 
                       domain: str,
                       concept: str,
                       context_samples: List[Dict],
                       batch_size: int) -> str:
        """Generate prompt for specific domain-concept pair."""
        context_str = "\n".join([
            f"Example {i+1}:\nAttribute: {s['attribute_name']}\nDescription: {s['description']}"
            for i, s in enumerate(context_samples)
        ])
        
        return f"""Based on these examples from {domain} - {concept}:
{context_str}

Generate {batch_size} new, unique analytics attributes.
Each must follow the pattern shown in examples but be distinctly different.

Requirements:
1. Attribute names must be in snake_case
2. Descriptions should be clear and concise
3. Must be different from examples
4. Follow the exact pattern seen in examples

Provide exactly {batch_size} responses in this format, one per line:
{{"attribute_name": "name", "description": "description"}}

Output {batch_size} lines of JSON only, no additional text."""

    def generate_batch(self,
                      domain: str,
                      concept: str,
                      context_samples: List[Dict],
                      batch_size: int,
                      used_samples: Set[str]) -> List[Dict]:
        """Generate a single batch with thread-local rate limiting."""
        llm_handler = self.get_llm_handler()
        context = random.sample(context_samples, min(CONTEXT_SIZE, len(context_samples)))
        prompt = self.generate_prompt(domain, concept, context, batch_size)
        
        for attempt in range(MAX_RETRIES):
            try:
                with self.stats_lock:
                    self.generation_stats['attempts'] += 1

                llm_handler.wait_if_needed()
                response = chinou_response(prompt)
                
                # Process response
                valid_samples = []
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                
                for line in lines:
                    try:
                        sample = json.loads(line)
                        if all(key in sample for key in ['attribute_name', 'description']):
                            sample['domain'] = domain
                            sample['concept'] = concept
                            sample_key = f"{sample['attribute_name']}_{sample['description']}"
                            
                            with self.used_samples_lock:
                                if sample_key not in used_samples:
                                    used_samples.add(sample_key)
                                    valid_samples.append(sample)
                    except json.JSONDecodeError:
                        continue

                if len(valid_samples) == batch_size:
                    llm_handler.adjust_delay(True)
                    with self.stats_lock:
                        self.generation_stats['successes'] += 1
                    return valid_samples

                llm_handler.adjust_delay(False)
                
            except Exception as e:
                llm_handler.adjust_delay(False)
                logger.warning(f"Batch generation attempt {attempt + 1} failed: {str(e)}")
                time.sleep(1)

        with self.stats_lock:
            self.generation_stats['failures'] += 1
        return []

    def generate_samples_parallel(self,
                                domain: str,
                                concept: str,
                                base_samples: List[Dict],
                                num_needed: int,
                                used_samples: Set[str]) -> List[Dict]:
        """Generate samples using parallel execution."""
        results = []
        num_batches = (num_needed + BATCH_SIZE - 1) // BATCH_SIZE
        tasks = []
        
        for _ in range(num_batches):
            batch_size = min(BATCH_SIZE, num_needed - len(results))
            if batch_size <= 0:
                break
            tasks.append((domain, concept, base_samples, batch_size, used_samples))
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for task in tasks:
                future = executor.submit(self.generate_batch, *task)
                futures.append(future)
            
            for future in tqdm(as_completed(futures), 
                             total=len(futures),
                             desc=f"Generating {domain}-{concept}"):
                batch_results = future.result()
                results.extend(batch_results)
                
                # Break if we have enough samples
                if len(results) >= num_needed:
                    break
        
        return results[:num_needed]

class DataBalancer:
    """Main class for parallel data balancing."""
    
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.generator = ParallelSampleGenerator(max_workers)

    def balance_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Balance dataset using parallel processing."""
        training_samples = []
        eval_samples = []
        used_samples = set()
        
        # Process each domain-concept pair
        for (domain, concept), group in df.groupby(['domain', 'concept']):
            samples = group.to_dict('records')
            logger.info(f"\nProcessing {domain}-{concept}: {len(samples)} samples")
            
            # Track existing samples
            for sample in samples:
                used_samples.add(f"{sample['attribute_name']}_{sample['description']}")
            
            if len(samples) > SAMPLES_PER_CLASS:
                # Move excess to evaluation
                random.shuffle(samples)
                training_samples.extend(samples[:SAMPLES_PER_CLASS])
                eval_samples.extend(samples[SAMPLES_PER_CLASS:])
            else:
                # Add existing and generate synthetic
                training_samples.extend(samples)
                num_needed = SAMPLES_PER_CLASS - len(samples)
                
                if num_needed > 0:
                    synthetic_samples = self.generator.generate_samples_parallel(
                        domain,
                        concept,
                        samples,
                        num_needed,
                        used_samples
                    )
                    training_samples.extend(synthetic_samples)
        
        return pd.DataFrame(training_samples), pd.DataFrame(eval_samples)

def main(input_path: str, output_dir: str = 'data', max_workers: int = MAX_WORKERS):
    """Main execution function."""
    try:
        # Setup
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read and validate input
        logger.info(f"Reading data from {input_path}")
        df = pd.read_csv(input_path)
        
        required_columns = {'attribute_name', 'description', 'domain', 'concept'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Input CSV must contain columns: {required_columns}")
        
        # Process data
        balancer = DataBalancer(max_workers)
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
        
        # Generate report
        stats = balancer.generator.generation_stats
        report = [
            "Data Balance Report",
            "=" * 50,
            f"\nProcessing Statistics:",
            f"Processing Time: {processing_time:.2f} seconds",
            f"Original Records: {len(df)}",
            f"Training Records: {len(training_df)}",
            f"Evaluation Records: {len(eval_df)}",
            f"Generated Records: {len(training_df) - (len(df) - len(eval_df))}",
            f"\nGeneration Attempts: {stats['attempts']}",
            f"Successful Generations: {stats['successes']}",
            f"Failed Generations: {stats['failures']}",
            f"Success Rate: {stats['successes']/stats['attempts']*100:.2f}%"
        ]
        
        report_path = output_path / 'balance_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info("Processing complete!")
        logger.info(f"Files saved in: {output_path}")
        
        return str(training_path), str(eval_path)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Balance dataset with parallel LLM processing')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, 
                       help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    
    try:
        main(args.input, args.output_dir, args.max_workers)
    except Exception as e:
        logger.error(f"Failed to process dataset: {str(e)}")
        raise
