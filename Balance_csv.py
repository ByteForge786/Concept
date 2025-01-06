import pandas as pd
import numpy as np
from pathlib import Path
import faiss
import pickle
from typing import List, Dict, Tuple, Set, Optional
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
class SynthesisTask:
    """Structure for synthesis tasks."""
    domain: str
    concept: str
    context_samples: List[Dict]
    num_samples: int

class ResponseValidator:
    """Validates LLM responses and ensures completeness."""
    
    @staticmethod
    def is_valid_json(response_line: str) -> bool:
        """Check if response line is valid JSON."""
        try:
            data = json.loads(response_line)
            required_keys = {'attribute_name', 'description'}
            return all(key in data for key in required_keys) and \
                   all(isinstance(data[key], str) for key in required_keys) and \
                   all(data[key].strip() for key in required_keys)
        except:
            return False
    
    @staticmethod
    def validate_batch_response(response: str, expected_count: int) -> List[Dict]:
        """Validate complete batch response."""
        valid_samples = []
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        for line in lines:
            if ResponseValidator.is_valid_json(line):
                valid_samples.append(json.loads(line))
        
        return valid_samples if len(valid_samples) == expected_count else []

class EmbeddingManager:
    """Manages embeddings and vector similarity search."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        self.embed_model = SentenceTransformer(model_name)
        self.index_dict = {}
        self.embeddings_dict = {}
        
    def create_embedding(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for texts."""
        return self.embed_model.encode(texts, show_progress_bar=False)
    
    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Build FAISS index for embeddings."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        if len(embeddings) > 0:
            index.add(embeddings.astype('float32'))
        return index
    
    def save_state(self, save_path: str):
        """Save embeddings and indexes."""
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
        
    def generate_prompt(self, domain: str, concept: str, context_samples: List[Dict]) -> str:
        """Generate prompt for LLM with specific domain-concept context."""
        context_str = "\n".join([
            f"Example {i+1}:\nAttribute: {s['attribute_name']}\nDescription: {s['description']}"
            for i, s in enumerate(context_samples)
        ])
        
        return f"""Based on these examples from {domain} - {concept}:
{context_str}

Generate {self.batch_size} new, unique analytics attributes.
Each must follow the pattern shown in examples but be distinctly different.

Requirements:
1. Attribute names must be in snake_case
2. Descriptions should be clear and concise
3. Must be different from examples
4. Follow the exact pattern seen in examples

Provide exactly {self.batch_size} responses in this format, one per line:
{{"attribute_name": "name", "description": "description"}}

Output {self.batch_size} lines of JSON only, no additional text."""
    
    def get_shuffled_context(self, samples: List[Dict], k: int) -> List[Dict]:
        """Get shuffled context samples."""
        available = samples.copy()
        random.shuffle(available)
        return available[:k]
    
    def generate_batch(self,
                      domain: str,
                      concept: str,
                      base_samples: List[Dict],
                      k: int,
                      used_samples: Set[str]) -> List[Dict]:
        """Generate a batch of samples with retry logic."""
        context = self.get_shuffled_context(base_samples, k)
        prompt = self.generate_prompt(domain, concept, context)
        
        for attempt in range(MAX_RETRIES):
            try:
                response = chinou_response(prompt)
                samples = ResponseValidator.validate_batch_response(
                    response, 
                    self.batch_size
                )
                
                if not samples:
                    continue
                
                # Add domain and concept, check uniqueness
                valid_samples = []
                with self.used_samples_lock:
                    for sample in samples:
                        sample['domain'] = domain
                        sample['concept'] = concept
                        sample_key = f"{sample['attribute_name']}_{sample['description']}"
                        
                        if sample_key not in used_samples:
                            used_samples.add(sample_key)
                            valid_samples.append(sample)
                
                if len(valid_samples) == self.batch_size:
                    return valid_samples
                    
            except Exception as e:
                logger.error(f"Batch generation attempt {attempt + 1} failed: {str(e)}")
                time.sleep(1)
        
        return []

class DataBalancer:
    """Main class for data balancing operations."""
    
    def __init__(self,
                 max_workers: int = MAX_WORKERS,
                 batch_size: int = BATCH_SIZE):
        self.max_workers = max_workers
        self.embedding_manager = EmbeddingManager()
        self.sample_generator = SampleGenerator(batch_size)
        
    def prepare_synthesis_tasks(self,
                              df: pd.DataFrame) -> Dict[str, SynthesisTask]:
        """Prepare synthesis tasks for each domain-concept pair."""
        tasks = {}
        
        for (domain, concept), group in df.groupby(['domain', 'concept']):
            samples = group.to_dict('records')
            
            if len(samples) >= SAMPLES_PER_CLASS:
                continue
                
            key = f"{domain}_{concept}"
            tasks[key] = SynthesisTask(
                domain=domain,
                concept=concept,
                context_samples=samples,
                num_samples=SAMPLES_PER_CLASS - len(samples)
            )
            
        return tasks
    
    def generate_samples_for_task(self,
                                task: SynthesisTask,
                                used_samples: Set[str]) -> List[Dict]:
        """Generate samples for a single synthesis task."""
        synthetic_samples = []
        k = min(CONTEXT_SIZE, len(task.context_samples))
        
        num_batches = (task.num_samples + self.sample_generator.batch_size - 1) \
                     // self.sample_generator.batch_size
                     
        for _ in range(num_batches):
            if len(synthetic_samples) >= task.num_samples:
                break
                
            batch = self.sample_generator.generate_batch(
                task.domain,
                task.concept,
                task.context_samples,
                k,
                used_samples
            )
            
            if batch:
                synthetic_samples.extend(batch)
            
        return synthetic_samples[:task.num_samples]
    
    def balance_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Balance the entire dataset."""
        logger.info("Starting dataset balancing...")
        
        # Initialize storage
        training_samples = []
        eval_samples = []
        used_samples = set()
        
        # Process existing samples
        for (domain, concept), group in df.groupby(['domain', 'concept']):
            samples = group.to_dict('records')
            
            if len(samples) > SAMPLES_PER_CLASS:
                # Move excess to evaluation
                random.shuffle(samples)
                training_samples.extend(samples[:SAMPLES_PER_CLASS])
                eval_samples.extend(samples[SAMPLES_PER_CLASS:])
            else:
                training_samples.extend(samples)
                
            # Track used samples
            for sample in samples:
                used_samples.add(f"{sample['attribute_name']}_{sample['description']}")
        
        # Prepare synthesis tasks
        synthesis_tasks = self.prepare_synthesis_tasks(df)
        
        # Generate synthetic samples in parallel
        if synthesis_tasks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(
                        self.generate_samples_for_task,
                        task,
                        used_samples
                    ): task_key
                    for task_key, task in synthesis_tasks.items()
                }
                
                for future in tqdm(as_completed(future_to_task),
                                 total=len(synthesis_tasks),
                                 desc="Generating synthetic samples"):
                    task_key = future_to_task[future]
                    try:
                        synthetic_samples = future.result()
                        training_samples.extend(synthetic_samples)
                        logger.info(f"Generated {len(synthetic_samples)} samples for {task_key}")
                    except Exception as e:
                        logger.error(f"Failed to generate samples for {task_key}: {str(e)}")
        
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
        
        # Generate report
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
            str(eval_df.groupby(['domain', 'concept']).size())
        ]
        
        # Save report
        report_path = output_path / 'balance_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"\nProcessing complete!")
        logger.info(f"Files saved:")
        logger.info(f"1. Training data: {training_path}")
        logger.info(f"2. Evaluation data: {eval_path}")
        logger.info(f"3. Balance report: {report_path}")
        
        return str(training_path), str(eval_path)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Balance dataset with RAG and parallel processing')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, help='Maximum number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for generation')
    
    args = parser.parse_args()
    
    try:
        main(
            args.input,
            args.output_dir,
            args.max_workers,
            args.batch_size
        )
    except Exception as e:
        logger.error(f"Failed to process dataset: {str(e)}")
        raise
