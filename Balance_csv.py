import pandas as pd
import numpy as np
from pathlib import Path
import faiss
import pickle
from typing import List, Dict, Tuple, Set, Optional, Iterator
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import random
from tqdm import tqdm
from test import *  # for chinou_response
import json
import logging
import time
from threading import Lock
import os
from itertools import islice
import math

# Constants
SAMPLES_PER_CLASS = 500
BATCH_SIZE = 5  # For LLM generation
EMBEDDING_BATCH_SIZE = 32  # For embedding creation
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
class SynthesisTask:
    """Structure for synthesis tasks."""
    domain: str
    concept: str
    context_samples: List[Dict]
    num_samples: int

class ParallelEmbeddingManager:
    """Manages parallel embedding creation and vector similarity search."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                 batch_size: int = EMBEDDING_BATCH_SIZE,
                 max_workers: int = MAX_WORKERS):
        self.embed_model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.index_dict = {}
        self.embeddings_dict = {}

    def batch_texts(self, texts: List[str]) -> Iterator[List[str]]:
        """Split texts into batches."""
        for i in range(0, len(texts), self.batch_size):
            yield texts[i:i + self.batch_size]

    def process_batch(self, batch: List[str]) -> np.ndarray:
        """Process a single batch of texts."""
        return self.embed_model.encode(batch, show_progress_bar=False)

    def create_embeddings_parallel(self, texts: List[str]) -> np.ndarray:
        """Create embeddings in parallel batches."""
        batches = list(self.batch_texts(texts))
        embeddings_list = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_batch, batch) 
                      for batch in batches]
            
            for future in tqdm(as_completed(futures),
                             total=len(futures),
                             desc="Creating embeddings"):
                try:
                    batch_embeddings = future.result()
                    embeddings_list.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error in embedding batch: {str(e)}")
                    raise

        # Combine all embeddings
        return np.vstack(embeddings_list) if embeddings_list else np.array([])

    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Build FAISS index for embeddings."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        if len(embeddings) > 0:
            index.add(embeddings.astype('float32'))
        return index

    def process_domain_concept(self, 
                             key: str,
                             samples: List[Dict]) -> Tuple[np.ndarray, faiss.IndexFlatL2]:
        """Process embeddings and index for a domain-concept pair."""
        texts = [f"{s['attribute_name']} {s['description']}" for s in samples]
        embeddings = self.create_embeddings_parallel(texts)
        index = self.build_index(embeddings)
        return embeddings, index

    def process_all_parallel(self, grouped_samples: Dict[str, List[Dict]]):
        """Process all domain-concept pairs in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_domain_concept, key, samples): key
                for key, samples in grouped_samples.items()
            }
            
            for future in tqdm(as_completed(futures),
                             total=len(futures),
                             desc="Processing domain-concepts"):
                key = futures[future]
                try:
                    embeddings, index = future.result()
                    self.embeddings_dict[key] = embeddings
                    self.index_dict[key] = index
                except Exception as e:
                    logger.error(f"Error processing {key}: {str(e)}")

    def save_state(self, save_path: str):
        """Save embeddings and indexes."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / 'embedding_state.pkl', 'wb') as f:
            pickle.dump({
                'index_dict': self.index_dict,
                'embeddings_dict': self.embeddings_dict
            }, f)

class ParallelSampleGenerator:
    """Handles parallel synthetic sample generation with context management."""
    
    def __init__(self, batch_size: int = BATCH_SIZE, max_workers: int = MAX_WORKERS):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.used_samples_lock = Lock()
    
    def generate_samples_parallel(self,
                                domain: str,
                                concept: str,
                                base_samples: List[Dict],
                                num_needed: int,
                                used_samples: Set[str]) -> List[Dict]:
        """Generate samples in parallel batches."""
        synthetic_samples = []
        k = min(CONTEXT_SIZE, len(base_samples))
        
        num_batches = math.ceil(num_needed / self.batch_size)
        remaining = num_needed
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for _ in range(num_batches):
                current_batch_size = min(self.batch_size, remaining)
                if current_batch_size <= 0:
                    break
                
                futures.append(
                    executor.submit(
                        self.generate_batch,
                        domain,
                        concept,
                        base_samples,
                        k,
                        current_batch_size,
                        used_samples
                    )
                )
                remaining -= current_batch_size
            
            for future in tqdm(as_completed(futures),
                             total=len(futures),
                             desc=f"Generating samples for {domain}-{concept}"):
                try:
                    batch_samples = future.result()
                    synthetic_samples.extend(batch_samples)
                except Exception as e:
                    logger.error(f"Batch generation error: {str(e)}")
        
        return synthetic_samples[:num_needed]

    def generate_batch(self,
                      domain: str,
                      concept: str,
                      base_samples: List[Dict],
                      k: int,
                      batch_size: int,
                      used_samples: Set[str]) -> List[Dict]:
        """Generate a single batch of samples."""
        context = self.get_shuffled_context(base_samples, k)
        prompt = self.generate_prompt(domain, concept, context, batch_size)
        
        for attempt in range(MAX_RETRIES):
            try:
                response = chinou_response(prompt)
                samples = self.validate_batch_response(response, batch_size)
                
                if not samples:
                    continue
                
                valid_samples = []
                with self.used_samples_lock:
                    for sample in samples:
                        sample['domain'] = domain
                        sample['concept'] = concept
                        sample_key = f"{sample['attribute_name']}_{sample['description']}"
                        
                        if sample_key not in used_samples:
                            used_samples.add(sample_key)
                            valid_samples.append(sample)
                
                if len(valid_samples) == batch_size:
                    return valid_samples
                    
            except Exception as e:
                logger.error(f"Batch generation attempt {attempt + 1} failed: {str(e)}")
                time.sleep(1)
        
        return []

    @staticmethod
    def get_shuffled_context(samples: List[Dict], k: int) -> List[Dict]:
        """Get shuffled context samples."""
        available = samples.copy()
        random.shuffle(available)
        return available[:k]

    @staticmethod
    def generate_prompt(domain: str, 
                       concept: str, 
                       context_samples: List[Dict],
                       batch_size: int) -> str:
        """Generate prompt for LLM."""
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
        """Validate LLM response."""
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

class ParallelDataBalancer:
    """Main class for parallel data balancing operations."""
    
    def __init__(self,
                 max_workers: int = MAX_WORKERS,
                 batch_size: int = BATCH_SIZE):
        self.max_workers = max_workers
        self.embedding_manager = ParallelEmbeddingManager(max_workers=max_workers)
        self.sample_generator = ParallelSampleGenerator(
            batch_size=batch_size,
            max_workers=max_workers
        )

    def prepare_tasks(self, df: pd.DataFrame) -> Dict[str, Tuple[str, str, List[Dict], int]]:
        """Prepare tasks for parallel processing."""
        tasks = {}
        
        for (domain, concept), group in df.groupby(['domain', 'concept']):
            samples = group.to_dict('records')
            key = f"{domain}_{concept}"
            
            if len(samples) < SAMPLES_PER_CLASS:
                tasks[key] = (
                    domain,
                    concept,
                    samples,
                    SAMPLES_PER_CLASS - len(samples)
                )
        
        return tasks

    def process_task(self,
                    domain: str,
                    concept: str,
                    samples: List[Dict],
                    num_needed: int,
                    used_samples: Set[str]) -> List[Dict]:
        """Process a single task."""
        return self.sample_generator.generate_samples_parallel(
            domain,
            concept,
            samples,
            num_needed,
            used_samples
        )

    def balance_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Balance dataset with parallel processing."""
        logger.info("Starting parallel dataset balancing...")
        
        # Initialize storage
        training_samples = []
        eval_samples = []
        used_samples = set()
        
        # Group samples by domain-concept
        grouped_samples = {}
        for (domain, concept), group in df.groupby(['domain', 'concept']):
            samples = group.to_dict('records')
            key = f"{domain}_{concept}"
            
            if len(samples) > SAMPLES_PER_CLASS:
                # Move excess to evaluation
                random.shuffle(samples)
                training_samples.extend(samples[:SAMPLES_PER_CLASS])
                eval_samples.extend(samples[SAMPLES_PER_CLASS:])
            else:
                training_samples.extend(samples)
                grouped_samples[key] = samples
            
            # Track used samples
            for sample in samples:
                used_samples.add(f"{sample['attribute_name']}_{sample['description']}")
        
        # Process embeddings in parallel
        logger.info("Processing embeddings...")
        self.embedding_manager.process_all_parallel(grouped_samples)
        
        # Prepare synthesis tasks
        tasks = self.prepare_tasks(df)
        
        # Generate synthetic samples in parallel
        if tasks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_task,
                        domain,
                        concept,
                        samples,
                        num_needed,
                        used_samples
                    ): key
                    for key, (domain, concept, samples, num_needed) in tasks.items()
                }
                
                for future in tqdm(as_completed(futures),
                                 total=len(futures),
                                 desc="Generating synthetic samples"):
                    key = futures[future]
                    try:
                        synthetic_samples = future.result()
                        training_samples.extend(synthetic_samples)
                        logger.info(f"Generated {len(synthetic_samples)} samples for {key}")
                    except Exception as e:
                        logger.error(f"Failed to generate samples for {key}: {str(e)}")
        
        # Create final dataframes
        training_df = pd.DataFrame(training_samples)
        eval_df = pd.DataFrame(eval_samples)
        
        # Save embeddings state
        self.embedding_manager.save_state('data/embeddings')
        
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
        
        # Read input data
        logger.info(f"Reading data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Initialize balancer
        balancer = ParallelDataBalancer(max_workers, batch_size)
        
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
            "\nTraining Set Distribution:",
            str(training_df.groupby(['domain', 'concept']).size()),
            "\nEvaluation Set Distribution:",
            str(eval_df.groupby(['domain', 'concept']).size()),
            f"\nProcessing Performance:",
            f"Average time per sample: {processing_time/len(training_df):.3f} seconds",
            f"Samples per second: {len(training_df)/processing_time:.2f}",
            f"\nParallelization Stats:",
            f"Worker processes: {max_workers}",
            f"Batch size (LLM): {batch_size}",
            f"Batch size (Embeddings): {EMBEDDING_BATCH_SIZE}"
        ]
        
        # Add synthetic samples stats
        synthetic_count = len(training_df) - (len(df) - len(eval_df))
        report.extend([
            f"\nSynthetic Sample Statistics:",
            f"Original samples: {len(df)}",
            f"Training samples: {len(training_df)}",
            f"Evaluation samples: {len(eval_df)}",
            f"Generated samples: {synthetic_count}",
            f"Generation ratio: {synthetic_count/len(df):.2f}x"
        ])
        
        # Save report
        report_path = output_path / 'balance_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"\nProcessing complete!")
        logger.info(f"Files saved:")
        logger.info(f"1. Training data: {training_path}")
        logger.info(f"2. Evaluation data: {eval_path}")
        logger.info(f"3. Balance report: {report_path}")
        logger.info(f"4. Embeddings state: {output_path}/embeddings/embedding_state.pkl")
        
        return str(training_path), str(eval_path)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Balance dataset with parallel processing and optimized embedding generation'
    )
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
                       help='Batch size for LLM generation')
    parser.add_argument('--embedding-batch-size',
                       type=int,
                       default=EMBEDDING_BATCH_SIZE,
                       help='Batch size for embedding generation')
    
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
