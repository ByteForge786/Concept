import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import time
from threading import Lock
import os
import math
from queue import Queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from collections import defaultdict
import threading
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import re

# Constants
SAMPLES_PER_CLASS = 500
BATCH_SIZE = 10  # Increased batch size for better throughput
MAX_RETRIES = 5
MAX_WORKERS = 8  # Increased worker count
MAX_CONCURRENT_REQUESTS = 8  # Increased concurrent requests

# Configure logging with colors
import colorlog

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger('balancer')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@dataclass
class GenerationStats:
    """Track generation statistics per domain-concept pair"""
    original_samples: int = 0
    needed_samples: int = 0
    generated_samples: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    duplicates_avoided: int = 0
    start_time: float = 0.0
    
    def log_progress(self, domain: str, concept: str):
        elapsed = time.time() - self.start_time
        rate = self.generated_samples / elapsed if elapsed > 0 else 0
        logger.info(
            f"\n{domain}-{concept} Progress:"
            f"\n  Original samples: {self.original_samples}"
            f"\n  Needed samples: {self.needed_samples}"
            f"\n  Generated so far: {self.generated_samples}"
            f"\n  Generation rate: {rate:.2f} samples/second"
            f"\n  Success rate: {(self.successful_generations/(self.successful_generations + self.failed_generations)*100):.1f}%"
            f"\n  Duplicates avoided: {self.duplicates_avoided}"
        )

class ConcurrentAttributeTracker:
    """Thread-safe attribute tracker with per domain-concept locking"""
    
    def __init__(self):
        self._sets = defaultdict(set)
        self._locks = defaultdict(threading.Lock)
    
    def add_if_not_exists(self, domain: str, concept: str, attributes: List[str]) -> List[bool]:
        """Batch check and add attributes, returns list of success flags"""
        key = (domain, concept)
        with self._locks[key]:
            results = []
            for attr in attributes:
                if attr in self._sets[key]:
                    results.append(False)
                else:
                    self._sets[key].add(attr)
                    results.append(True)
            return results
    
    def get_used_attributes(self, domain: str, concept: str) -> Set[str]:
        key = (domain, concept)
        with self._locks[key]:
            return self._sets[key].copy()

class AdaptiveRateLimiter:
    """Improved rate limiter with adaptive delays"""
    
    def __init__(self, initial_delay: float = 0.1):
        self.delay = initial_delay
        self.min_delay = 0.05
        self.max_delay = 2.0
        self.success_window = []
        self.window_size = 10
        self.lock = Lock()
        
    def update_delay(self, success: bool):
        with self.lock:
            self.success_window.append(success)
            if len(self.success_window) > self.window_size:
                self.success_window.pop(0)
            
            success_rate = sum(self.success_window) / len(self.success_window)
            
            if success_rate > 0.8:  # Increase throughput
                self.delay = max(self.min_delay, self.delay * 0.9)
            elif success_rate < 0.5:  # Reduce throughput
                self.delay = min(self.max_delay, self.delay * 1.2)
    
    def wait(self):
        if self.delay > 0:
            time.sleep(self.delay)

class ImprovedSampleGenerator:
    """Enhanced sample generator with better parallelization"""
    
    def __init__(self, batch_size: int = BATCH_SIZE):
        self.batch_size = batch_size
        self.attribute_tracker = ConcurrentAttributeTracker()
        self.rate_limiter = AdaptiveRateLimiter()
        self.stats = defaultdict(GenerationStats)
    
    def generate_prompt(self, 
                       domain: str,
                       concept: str,
                       context_samples: List[Dict],
                       batch_size: int,
                       definition: str = "") -> str:
        # Create stronger context with examples
        context_str = "\n".join([
            f"Example {i+1}:\nAttribute: {s['attribute_name']}\nDescription: {s['description']}"
            for i, s in enumerate(context_samples)
        ])
        
        # Add definition if available
        definition_str = f"\nDefinition of {domain}-{concept}: {definition}\n" if definition else ""
        
        # Get currently used attributes
        used_attrs = self.attribute_tracker.get_used_attributes(domain, concept)
        avoid_str = "\nAvoid these existing attributes:\n" + ", ".join(used_attrs) if used_attrs else ""
        
        return f"""Generate {batch_size} unique analytics attributes for {domain} - {concept}.{definition_str}

Context examples:{avoid_str}
{context_str}

Requirements:
1. Attribute names must be in snake_case
2. Descriptions should be clear and specific
3. Each attribute must be unique and different from examples
4. Focus on {domain}-{concept} specific metrics and properties

Provide exactly {batch_size} responses in this format, one per line:
{{"attribute_name": "name", "description": "description"}}"""

    def validate_batch(self,
                      response: str,
                      domain: str,
                      concept: str) -> List[Dict]:
        """Validate and process a batch of generated samples"""
        try:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            samples = []
            
            for line in lines:
                try:
                    sample = json.loads(line)
                    if not all(k in sample for k in ['attribute_name', 'description']):
                        continue
                        
                    # Validate snake_case
                    if not re.match(r'^[a-z][a-z0-9_]*$', sample['attribute_name']):
                        continue
                        
                    samples.append(sample)
                except:
                    continue
            
            # Batch check for uniqueness
            attr_names = [s['attribute_name'] for s in samples]
            valid_flags = self.attribute_tracker.add_if_not_exists(domain, concept, attr_names)
            
            valid_samples = []
            stats = self.stats[(domain, concept)]
            
            for sample, is_valid in zip(samples, valid_flags):
                if is_valid:
                    sample['domain'] = domain
                    sample['concept'] = concept
                    valid_samples.append(sample)
                else:
                    stats.duplicates_avoided += 1
            
            return valid_samples
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return []

    async def generate_batch(self,
                           domain: str,
                           concept: str,
                           context_samples: List[Dict],
                           definition: str = "") -> List[Dict]:
        """Generate a single batch of samples"""
        prompt = self.generate_prompt(domain, concept, context_samples, self.batch_size, definition)
        
        try:
            self.rate_limiter.wait()
            response = chinou_response(prompt)  # Replace with your actual LLM call
            self.rate_limiter.update_delay(True)
            
            samples = self.validate_batch(response, domain, concept)
            stats = self.stats[(domain, concept)]
            
            if samples:
                stats.successful_generations += 1
                stats.generated_samples += len(samples)
            else:
                stats.failed_generations += 1
            
            return samples
            
        except Exception as e:
            logger.error(f"Batch generation error: {str(e)}")
            self.rate_limiter.update_delay(False)
            self.stats[(domain, concept)].failed_generations += 1
            return []

    def generate_samples(self,
                        domain: str,
                        concept: str,
                        base_samples: List[Dict],
                        num_needed: int,
                        definition: str = "") -> List[Dict]:
        """Generate samples with improved parallel processing"""
        
        # Initialize stats
        stats = self.stats[(domain, concept)]
        stats.original_samples = len(base_samples)
        stats.needed_samples = num_needed
        stats.start_time = time.time()
        
        logger.info(f"\nStarting generation for {domain}-{concept}")
        logger.info(f"Original samples: {len(base_samples)}")
        logger.info(f"Samples needed: {num_needed}")
        
        synthetic_samples = []
        context_size = min(5, len(base_samples))
        
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            futures = []
            
            while len(synthetic_samples) < num_needed:
                # Calculate remaining samples
                remaining = num_needed - len(synthetic_samples)
                num_batches = math.ceil(remaining / self.batch_size)
                
                # Submit batch requests
                for _ in range(num_batches):
                    context = random.sample(base_samples, context_size)
                    future = executor.submit(
                        self.generate_batch,
                        domain,
                        concept,
                        context,
                        definition
                    )
                    futures.append(future)
                
                # Process completed futures
                for future in as_completed(futures):
                    batch = future.result()
                    if batch:
                        synthetic_samples.extend(batch)
                        
                    # Log progress
                    if len(synthetic_samples) % 50 == 0:
                        stats.log_progress(domain, concept)
                    
                    if len(synthetic_samples) >= num_needed:
                        break
                
                # Clean up futures
                futures = [f for f in futures if not f.done()]
        
        # Final progress log
        stats.log_progress(domain, concept)
        return synthetic_samples[:num_needed]

class ImprovedDataBalancer:
    """Main class for improved data balancing operations"""
    
    def __init__(self,
                 batch_size: int = BATCH_SIZE,
                 domain_defs_path: Optional[str] = None):
        self.sample_generator = ImprovedSampleGenerator(batch_size)
        self.domain_defs = self._load_definitions(domain_defs_path) if domain_defs_path else {}
    
    def _load_definitions(self, path: str) -> Dict[str, str]:
        definitions = {}
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        domain, concept, definition = line.strip().split('|')
                        definitions[(domain.strip(), concept.strip())] = definition.strip()
        except Exception as e:
            logger.error(f"Error loading definitions: {str(e)}")
        return definitions
    
    def balance_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Balance dataset with improved logging and processing"""
        logger.info("\nStarting dataset balancing...")
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Unique domain-concept pairs: {df.groupby(['domain', 'concept']).size().shape[0]}")
        
        start_time = time.time()
        training_samples = []
        eval_samples = []
        
        # Process each domain-concept pair
        for (domain, concept), group in df.groupby(['domain', 'concept']):
            samples = group.to_dict('records')
            definition = self.domain_defs.get((domain, concept), "")
            
            logger.info(f"\nProcessing {domain}-{concept}")
            logger.info(f"Original samples: {len(samples)}")
            
            if len(samples) > SAMPLES_PER_CLASS:
                # Split into training and eval
                random.shuffle(samples)
                training_samples.extend(samples[:SAMPLES_PER_CLASS])
                eval_samples.extend(samples[SAMPLES_PER_CLASS:])
                logger.info(f"Split: {SAMPLES_PER_CLASS} training, {len(samples)-SAMPLES_PER_CLASS} evaluation")
            else:
                # Generate synthetic samples
                training_samples.extend(samples)
                num_needed = SAMPLES_PER_CLASS - len(samples)
                
                if num_needed > 0:
                    synthetic = self.sample_generator.generate_samples(
                        domain,
                        concept,
                        samples,
                        num_needed,
                        definition
                    )
                    training_samples.extend(synthetic)
        
        # Create final dataframes
        training_df = pd.DataFrame(training_samples)
        eval_df = pd.DataFrame(eval_samples)
        
        # Log final statistics
        total_time = time.time() - start_time
        logger.info("\nProcessing complete!")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Training samples: {len(training_df)}")
        logger.info(f"Evaluation samples: {len(eval_df)}")
        
        return training_df, eval_df

def main(input_path: str,
         output_dir: str = 'data',
         batch_size: int = BATCH_SIZE,
         domain_defs_path: Optional[str] = None):
    """Main execution function"""
    try:
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Add file logging
        file_handler = logging.FileHandler(output_path / 'balancer.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Read and validate input data
        logger.info(f"Reading data from {input_path}")
        df = pd.read_csv(input_path)
        
        required_columns = {'attribute_name', 'description', 'domain', 'concept'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Input CSV must contain columns: {required_columns}")
        
        # Initialize balancer
        balancer = ImprovedDataBalancer(batch_size, domain_defs_path)
        
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
        if not eval_df.empty:
            eval_df.to_csv(eval_path, index=False)
        
        # Generate comprehensive report
        report = [
            "Data Balance Report",
            "=" * 50,
            f"\nProcessing Configuration:",
            f"Batch Size: {batch_size}",
            f"Max Concurrent Requests: {MAX_CONCURRENT_REQUESTS}",
            f"Domain Definitions: {'Used' if domain_defs_path else 'Not Used'}",
            f"Processing Time: {processing_time:.2f} seconds",
            f"\nData Statistics:",
            f"Original Records: {len(df)}",
            f"Training Records: {len(training_df)}",
            f"Evaluation Records: {len(eval_df)}",
            f"Domains: {df['domain'].nunique()}",
            f"Concepts: {df['concept'].nunique()}",
            "\nOriginal Distribution:",
            str(df.groupby(['domain', 'concept']).size()),
            "\nBalanced Training Set Distribution:",
            str(training_df.groupby(['domain', 'concept']).size()),
            "\nDetailed Generation Statistics:"
        ]
        
        # Add per-domain-concept statistics
        for (domain, concept), stats in balancer.sample_generator.stats.items():
            total_attempts = stats.successful_generations + stats.failed_generations
            success_rate = (stats.successful_generations / total_attempts * 100) if total_attempts > 0 else 0
            
            report.extend([
                f"\n{domain}-{concept}:",
                f"  Original samples: {stats.original_samples}",
                f"  Generated samples: {stats.generated_samples}",
                f"  Success rate: {success_rate:.1f}%",
                f"  Duplicates avoided: {stats.duplicates_avoided}"
            ])
        
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
        
        return str(training_path), str(eval_path)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Balance dataset with improved parallelization')
    parser.add_argument('--input', 
                       required=True,
                       help='Input CSV file path')
    parser.add_argument('--output-dir',
                       default='data',
                       help='Output directory')
    parser.add_argument('--batch-size',
                       type=int,
                       default=BATCH_SIZE,
                       help='Batch size for generation')
    parser.add_argument('--domain-defs',
                       help='Path to domain-concept definitions file')
    
    args = parser.parse_args()
    
    try:
        main(
            args.input,
            args.output_dir,
            args.batch_size,
            args.domain_defs
        )
        logger.info("Processing completed successfully!")
    except Exception as e:
        logger.error(f"Failed to process dataset: {str(e)}")
        raise
