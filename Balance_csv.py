import pandas as pd
import numpy as np
from pathlib import Path
import faiss
import pickle
from typing import List, Dict, Tuple, Set, Optional, Callable, Any, Iterator
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from tqdm import tqdm
from test import *  # for chinou_response
import json
import logging
import time
from threading import Lock
import os
import math

# Constants
SAMPLES_PER_CLASS = 500
BATCH_SIZE = 5
EMBEDDING_BATCH_SIZE = 32
MAX_RETRIES = 5
MAX_WORKERS = 4
CONTEXT_SIZE = 6

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdaptiveRateLimiter:
    """Adapts to API limits by monitoring responses."""
    
    def __init__(self):
        self.lock = Lock()
        self.error_window = []
        self.window_size = 10
        self.current_delay = 1.0
        self.min_delay = 0.5
        self.max_delay = 30.0
        self.last_request_time = 0
        self.success_count = 0
        self.error_count = 0

    def update_delay(self, success: bool):
        with self.lock:
            self.error_window.append(not success)
            if len(self.error_window) > self.window_size:
                self.error_window.pop(0)
            
            error_rate = sum(self.error_window) / len(self.error_window)
            
            if success:
                self.success_count += 1
                if self.success_count >= 5:
                    self.current_delay = max(
                        self.min_delay,
                        self.current_delay * 0.9
                    )
                    self.success_count = 0
            else:
                self.error_count += 1
                self.success_count = 0
                self.current_delay = min(
                    self.max_delay,
                    self.current_delay * 2
                )
            
            if error_rate > 0.1:  # More than 10% errors
                logger.warning(f"High error rate detected: {error_rate:.2%}")

    def wait(self):
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            wait_time = max(0, self.current_delay - time_since_last)
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request_time = time.time()

class AdaptiveRetryHandler:
    """Handles retries with adaptive backoff."""
    
    def __init__(self, max_retries: int = MAX_RETRIES):
        self.max_retries = max_retries
        self.rate_limiter = AdaptiveRateLimiter()

    def execute_with_retry(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.wait()
                result = func(*args, **kwargs)
                self.rate_limiter.update_delay(success=True)
                return result
                
            except Exception as e:
                last_exception = e
                self.rate_limiter.update_delay(success=False)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}. "
                    f"Current delay: {self.rate_limiter.current_delay:.2f}s"
                )
                
        raise last_exception

@dataclass
class GenerationTask:
    """Structure for generation tasks."""
    domain: str
    concept: str
    context_samples: List[Dict]
    num_samples: int

class ParallelEmbeddingManager:
    """Manages parallel embedding creation and vector similarity search."""
    
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                 batch_size: int = EMBEDDING_BATCH_SIZE,
                 max_workers: int = MAX_WORKERS):
        self.embed_model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.index_dict = {}
        self.embeddings_dict = {}

    def batch_texts(self, texts: List[str]) -> Iterator[List[str]]:
        for i in range(0, len(texts), self.batch_size):
            yield texts[i:i + self.batch_size]

    def process_batch(self, batch: List[str]) -> np.ndarray:
        return self.embed_model.encode(batch, show_progress_bar=False)

    def create_embeddings_parallel(self, texts: List[str]) -> np.ndarray:
        batches = list(self.batch_texts(texts))
        embeddings_list = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_batch, batch) 
                      for batch in batches]
            
            for future in tqdm(as_completed(futures),
                             total=len(futures),
                             desc="Creating embeddings"):
                embeddings_list.append(future.result())

        return np.vstack(embeddings_list) if embeddings_list else np.array([])

    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        if len(embeddings) > 0:
            index.add(embeddings.astype('float32'))
        return index

    def process_domain_concept(self, 
                             key: str,
                             samples: List[Dict]) -> Tuple[np.ndarray, faiss.IndexFlatL2]:
        texts = [f"{s['attribute_name']} {s['description']}" for s in samples]
        embeddings = self.create_embeddings_parallel(texts)
        index = self.build_index(embeddings)
        return embeddings, index

    def save_state(self, save_path: str):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / 'embedding_state.pkl', 'wb') as f:
            pickle.dump({
                'index_dict': self.index_dict,
                'embeddings_dict': self.embeddings_dict
            }, f)

class SampleGenerator:
    """Handles synthetic sample generation with context management."""
    
    def __init__(self, batch_size: int = BATCH_SIZE):
        self.batch_size = batch_size
        self.used_samples_lock = Lock()
        self.retry_handler = AdaptiveRetryHandler()
        self.generation_stats = {
            'attempts': 0,
            'successes': 0,
            'failures': 0
        }

    def _determine_k(self, sample_size: int) -> int:
        """Determine dynamic k based on sample size."""
        if sample_size <= 1:
            return 1
        elif sample_size <= 3:
            return min(sample_size, 2)
        elif sample_size <= 10:
            return min(sample_size, 3)
        else:
            return min(sample_size, 6)

    def get_shuffled_context(self, samples: List[Dict], k: int) -> List[Dict]:
        """Get shuffled context samples."""
        available = samples.copy()
        random.shuffle(available)
        return available[:k]

    def generate_prompt(self, 
                       domain: str,
                       concept: str,
                       context_samples: List[Dict],
                       batch_size: int) -> str:
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

    @staticmethod
    def validate_batch_response(response: str, expected_count: int) -> List[Dict]:
        valid_samples = []
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        for line in lines:
            try:
                sample = json.loads(line)
                if all(key in sample for key in ['attribute_name', 'description']):
                    valid_samples.append(sample)
            except:
                continue
        
        return valid_samples if len(valid_samples) == expected_count else []

    def generate_batch(self,
                      domain: str,
                      concept: str,
                      base_samples: List[Dict],
                      k: int,
                      used_samples: Set[str]) -> List[Dict]:
        """Generate batch with retry and rate limiting."""
        context = self.get_shuffled_context(base_samples, k)
        prompt = self.generate_prompt(domain, concept, context, self.batch_size)
        
        def _make_llm_call():
            self.generation_stats['attempts'] += 1
            return chinou_response(prompt)
        
        try:
            response = self.retry_handler.execute_with_retry(_make_llm_call)
            samples = self.validate_batch_response(response, self.batch_size)
            
            if not samples:
                self.generation_stats['failures'] += 1
                return []
            
            valid_samples = []
            with self.used_samples_lock:
                for sample in samples:
                    sample['domain'] = domain
                    sample['concept'] = concept
                    sample_key = f"{sample['attribute_name']}_{sample['description']}"
                    
                    if sample_key not in used_samples:
                        used_samples.add(sample_key)
                        valid_samples.append(sample)
            
            if valid_samples:
                self.generation_stats['successes'] += 1
                
            return valid_samples
                
        except Exception as e:
            self.generation_stats['failures'] += 1
            logger.error(f"Batch generation failed after all retries: {str(e)}")
            return []

    def generate_samples(self,
                        domain: str,
                        concept: str,
                        base_samples: List[Dict],
                        num_needed: int,
                        used_samples: Set[str]) -> List[Dict]:
        """Generate all needed samples for a domain-concept pair."""
        synthetic_samples = []
        k = self._determine_k(len(base_samples))
        
        while len(synthetic_samples) < num_needed:
            batch = self.generate_batch(
                domain,
                concept,
                base_samples,
                k,
                used_samples
            )
            
            if batch:
                synthetic_samples.extend(batch)
            else:
                logger.warning(f"Failed to generate batch for {domain}-{concept}")
                time.sleep(2)  # Additional cooldown on failure
        
        return synthetic_samples[:num_needed]

    def log_generation_stats(self):
        stats = self.generation_stats
        success_rate = (stats['successes'] / stats['attempts']) if stats['attempts'] > 0 else 0
        
        logger.info("\nGeneration Statistics:")
        logger.info(f"Total attempts: {stats['attempts']}")
        logger.info(f"Successful generations: {stats['successes']}")
        logger.info(f"Failed generations: {stats['failures']}")
        logger.info(f"Success rate: {success_rate:.2%}")
        logger.info(f"Current delay: {self.retry_handler.rate_limiter.current_delay:.2f}s")

class DataBalancer:
    """Main class for data balancing operations."""
    
    def __init__(self,
                 max_workers: int = MAX_WORKERS,
                 batch_size: int = BATCH_SIZE):
        self.max_workers = max_workers
        self.sample_generator = SampleGenerator(batch_size)
        self.embedding_manager = ParallelEmbeddingManager(max_workers=max_workers)

    def balance_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Balance dataset with parallel processing."""
        logger.info("Starting dataset balancing...")
        
        # Initialize storage
        training_samples = []
        eval_samples = []
        used_samples = set()
        
        # Process each domain-concept pair
        for (domain, concept), group in df.groupby(['domain', 'concept']):
            samples = group.to_dict('records')
            logger.info(f"\nProcessing {domain}-{concept}: {len(samples)} samples")
            
            if len(samples) > SAMPLES_PER_CLASS:
                # Move excess to evaluation
                random.shuffle(samples)
                training_samples.extend(samples[:SAMPLES_PER_CLASS])
                eval_samples.extend(samples[SAMPLES_PER_CLASS:])
                logger.info(f"Split into {SAMPLES_PER_CLASS} training and {len(samples)-SAMPLES_PER_CLASS} evaluation samples")
            else:
                # Add all to training and generate synthetic
                training_samples.extend(samples)
                num_needed = SAMPLES_PER_CLASS - len(samples)
                
                if num_needed > 0:
                    logger.info(f"Generating {num_needed} synthetic samples...")
                    synthetic = self.sample_generator.generate_samples(
                        domain,
                        concept,
                        samples,
                        num_needed,
                        used_samples
                    )
                    training_samples.extend(synthetic)
                    logger.info(f"Generated {len(synthetic)} synthetic samples")
            
            # Track used samples
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
