import os
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Generator
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import joblib
from pathlib import Path
import yaml
from tqdm import tqdm
import random
from itertools import combinations, product
import json
from datetime import datetime
import time
from functools import lru_cache, partial
from collections import defaultdict, deque
import multiprocessing
from test import *

# Configure logging
log_path = Path('logs')
log_path.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

[Previous rate limiter code remains the same...]

@dataclass
class Config:
    """Configuration class for model training and prediction."""
    description_refine: bool
    classification_head: str
    embedding_model: str = 'sentence-transformers/all-mpnet-base-v2'
    batch_size: int = 32
    num_epochs: int = 1  # Changed to 1 as requested
    test_size: float = 0.2
    hard_negative_ratio: float = 0.5  # New parameter for hard negative sampling
    evaluation_steps: int = 500  # Added explicit parameter
    max_workers: int = min(32, multiprocessing.cpu_count() * 2)
    random_state: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.domain_encoder = LabelEncoder()
        self.concept_encoder = LabelEncoder()
        self.model = SentenceTransformer(config.embedding_model).to(config.device)
        
    def process_descriptions_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Highly parallel description processing using ProcessPoolExecutor."""
        if not self.config.description_refine:
            return df
            
        logger.info("Starting parallel description standardization...")
        
        # Convert to numpy arrays and create batches
        attr_names = df['attribute_name'].values
        descriptions = df['description'].values
        total_samples = len(df)
        
        # Optimize batch size based on CPU count
        optimal_batch_size = max(1, total_samples // (self.config.max_workers * 4))
        batches = [
            (attr_names[i:i+optimal_batch_size], descriptions[i:i+optimal_batch_size])
            for i in range(0, total_samples, optimal_batch_size)
        ]
        
        standardized_descriptions = []
        
        # Process batches in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for batch_attrs, batch_descs in batches:
                future = executor.submit(
                    self._process_description_batch,
                    batch_attrs,
                    batch_descs
                )
                futures.append(future)
            
            # Collect results maintaining order
            for future in tqdm(as_completed(futures), 
                             total=len(futures),
                             desc="Processing description batches"):
                try:
                    batch_results = future.result()
                    standardized_descriptions.extend(batch_results)
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    raise
        
        df['processed_description'] = standardized_descriptions
        return df
    
    @staticmethod
    def _process_description_batch(attr_names: np.ndarray, descriptions: np.ndarray) -> List[str]:
        """Process a batch of descriptions with retry mechanism."""
        results = []
        rate_limiter = AdaptiveRateLimiter()  # Local rate limiter for the process
        
        for attr_name, desc in zip(attr_names, descriptions):
            try:
                rate_limiter.wait()
                response = chinou_response(f"""Please standardize this attribute description to match analytics metric pattern:
                Attribute: {attr_name}
                Description: {desc}
                Output a single standardized description focused on measurement and purpose.""")
                rate_limiter.success()
                results.append(response)
            except Exception as e:
                logger.warning(f"Failed to refine description: {e}")
                rate_limiter.failure()
                results.append(desc)
        
        return results

    def create_pairs(self, df: pd.DataFrame) -> List[InputExample]:
        """Create training pairs using hierarchical strategy with parallel processing."""
        logger.info("Creating hierarchical training pairs...")
        
        data = df.copy()
        data['text'] = data['processed_description'] if 'processed_description' in df.columns else data['description']
        
        # Organize data by domain and concept
        domain_groups = data.groupby('domain')
        
        # Parallel pair generation
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Generate positive pairs
            futures_positive = []
            for domain_name, domain_data in domain_groups:
                future = executor.submit(
                    self._generate_positive_pairs,
                    domain_data
                )
                futures_positive.append(future)
            
            positive_pairs = []
            for future in tqdm(as_completed(futures_positive), 
                             total=len(futures_positive),
                             desc="Generating positive pairs"):
                positive_pairs.extend(future.result())
            
            # Generate hard negative pairs (same domain, different concept)
            futures_hard_neg = []
            for domain_name, domain_data in domain_groups:
                future = executor.submit(
                    self._generate_hard_negative_pairs,
                    domain_data,
                    self.config.hard_negative_ratio
                )
                futures_hard_neg.append(future)
            
            hard_negative_pairs = []
            for future in tqdm(as_completed(futures_hard_neg), 
                             total=len(futures_hard_neg),
                             desc="Generating hard negative pairs"):
                hard_negative_pairs.extend(future.result())
        
        # Combine and shuffle
        all_pairs = positive_pairs + hard_negative_pairs
        random.shuffle(all_pairs)
        
        logger.info(f"Created {len(positive_pairs)} positive pairs and {len(hard_negative_pairs)} hard negative pairs")
        return all_pairs

    @staticmethod
    def _generate_positive_pairs(domain_data: pd.DataFrame) -> List[InputExample]:
        """Generate positive pairs within each concept."""
        positive_pairs = []
        for _, concept_data in domain_data.groupby('concept'):
            texts = concept_data['text'].tolist()
            for text1, text2 in combinations(texts, 2):
                positive_pairs.append(InputExample(texts=[text1, text2], label=1.0))
        return positive_pairs

    @staticmethod
    def _generate_hard_negative_pairs(domain_data: pd.DataFrame, ratio: float) -> List[InputExample]:
        """Generate hard negative pairs within same domain but different concepts."""
        hard_negative_pairs = []
        concepts = domain_data.groupby('concept')['text'].apply(list).to_dict()
        
        for concept1, concept2 in combinations(concepts.keys(), 2):
            # Calculate number of pairs based on ratio
            num_pairs = int(min(len(concepts[concept1]), len(concepts[concept2])) * ratio)
            if num_pairs > 0:
                # Sample texts from each concept
                texts1 = random.sample(concepts[concept1], min(len(concepts[concept1]), num_pairs))
                texts2 = random.sample(concepts[concept2], min(len(concepts[concept2]), num_pairs))
                
                # Create pairs
                for text1, text2 in zip(texts1, texts2):
                    hard_negative_pairs.append(InputExample(texts=[text1, text2], label=0.0))
        
        return hard_negative_pairs

[Previous EmbeddingTrainer class with updated configuration...]
class EmbeddingTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = SentenceTransformer(config.embedding_model)
        self.model.to(self.device)
        
    def train(self, train_pairs: List[InputExample], 
              val_pairs: List[InputExample], 
              output_path: str):
        logger.info("Starting embedding model training...")
        
        os.makedirs(output_path, exist_ok=True)
        
        train_dataloader = DataLoader(
            train_pairs, 
            shuffle=True, 
            batch_size=self.config.batch_size,
            pin_memory=True if self.device.type == 'cuda' else False,
            num_workers=self.config.max_workers
        )
        
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
            val_pairs, 
            batch_size=self.config.batch_size,
            name='validation',
            show_progress_bar=True
        )
        
        warmup_steps = int(len(train_dataloader) * 0.1)
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=self.config.num_epochs,
            evaluation_steps=self.config.evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=output_path,
            save_best_model=True,
            show_progress_bar=True,
            use_amp=True if self.device.type == 'cuda' else False
        )
        
        return self.model

[Rest of the classes remain the same with their existing optimizations...]

def main():
    # Main function implementation remains the same
    pass

if __name__ == "__main__":
    main()
