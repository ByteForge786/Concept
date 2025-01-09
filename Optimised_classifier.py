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
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
from pathlib import Path
import yaml
from tqdm import tqdm
import random
from itertools import combinations
import json
from datetime import datetime
import time
from functools import lru_cache
from collections import defaultdict, deque
from test import *  # Import chinou_response from test module

# Configure logging
log_path = Path('logs')
log_path.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Adaptive rate limiter for API calls
class AdaptiveRateLimiter:
    def __init__(self, initial_rate: float = 1.0, window_size: int = 10):
        self.window_size = window_size
        self.initial_rate = initial_rate
        self.current_rate = initial_rate
        self.last_requests = deque(maxlen=window_size)
        self.backoff_factor = 2
        self.recovery_factor = 1.2
        self.min_wait_time = 0.1
        self.max_wait_time = 60
        
    def wait(self):
        now = time.time()
        if self.last_requests:
            time_since_last = now - self.last_requests[-1]
            wait_time = max(self.min_wait_time, 1 / self.current_rate - time_since_last)
            if wait_time > 0:
                time.sleep(wait_time)
        
        self.last_requests.append(now)
        
    def success(self):
        """Call after successful request"""
        self.current_rate = min(self.current_rate * self.recovery_factor, self.initial_rate)
        
    def failure(self):
        """Call after failed request"""
        self.current_rate = max(self.min_wait_time, self.current_rate / self.backoff_factor)

# Global rate limiter instance
rate_limiter = AdaptiveRateLimiter()

@dataclass
class Config:
    """Configuration class for model training and prediction."""
    description_refine: bool
    classification_head: str
    embedding_model: str = 'sentence-transformers/all-mpnet-base-v2'
    batch_size: int = 32
    num_epochs: int = 3
    test_size: float = 0.2
    same_section_ratio: float = 1.0
    other_section_ratio: float = 0.3
    max_workers: int = 4
    random_state: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class DataProcessor:
    """Handles data processing and pair creation for training."""
    
    def __init__(self, config: Config):
        self.config = config
        self.domain_encoder = LabelEncoder()
        self.concept_encoder = LabelEncoder()
        self.model = SentenceTransformer(config.embedding_model).to(config.device)
        
    def retry_with_backoff(self, func, max_retries: int = 5, initial_wait: float = 1.0):
        """Retry function with exponential backoff."""
        retries = 0
        while True:
            try:
                rate_limiter.wait()
                result = func()
                rate_limiter.success()
                return result
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    logger.error(f"Max retries reached: {e}")
                    raise
                rate_limiter.failure()
                wait_time = initial_wait * (2 ** (retries - 1))
                logger.warning(f"Retry {retries}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
    
    def standardize_description_batch(self, batch: List[Tuple[str, str]]) -> List[str]:
        """Standardize a batch of descriptions using LLM with retry mechanism."""
        standardized = []
        for attr_name, desc in batch:
            try:
                std_desc = self.retry_with_backoff(
                    lambda: chinou_response(f"""Please standardize this attribute description to match analytics metric pattern:
                    Attribute: {attr_name}
                    Description: {desc}
                    Output a single standardized description focused on measurement and purpose.""")
                )
                standardized.append(std_desc)
            except Exception as e:
                logger.error(f"Error standardizing description: {e}")
                standardized.append(desc)
        return standardized

    def process_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process descriptions in optimized parallel batches."""
        if not self.config.description_refine:
            return df
            
        logger.info("Starting description standardization...")
        
        # Convert to numpy arrays for faster slicing
        attr_names = df['attribute_name'].values
        descriptions = df['description'].values
        
        # Create batches using numpy operations
        batch_size = 50
        n_samples = len(df)
        indices = np.arange(0, n_samples, batch_size)
        batches = [
            list(zip(attr_names[i:i+batch_size], descriptions[i:i+batch_size]))
            for i in indices
        ]
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self.standardize_description_batch, batch)
                for batch in batches
            ]
            
            standardized_descriptions = []
            for future in tqdm(as_completed(futures), total=len(futures)):
                standardized_descriptions.extend(future.result())
                
        df['processed_description'] = standardized_descriptions
        return df

    def create_pairs(self, df: pd.DataFrame) -> List[InputExample]:
        """Create training pairs with optimized processing."""
        # Use vectorized operations to create main_dict
        data = df.copy()
        data['text'] = data['processed_description'] if 'processed_description' in df.columns else data['description']
        
        # Group data efficiently
        grouped = data.groupby(['domain', 'concept'])['text'].apply(list).to_dict()
        main_dict = defaultdict(dict)
        for (domain, concept), texts in grouped.items():
            main_dict[domain][concept] = texts

        # Pre-allocate lists for better memory efficiency
        positive_pairs = []
        negative_pairs = []
        
        # Create positive pairs efficiently
        for domain in main_dict:
            for concept in main_dict[domain]:
                texts = main_dict[domain][concept]
                # Use numpy for combinations
                text_indices = np.array(list(combinations(range(len(texts)), 2)))
                if len(text_indices) > 0:
                    pos_pairs = [(texts[i], texts[j], 1.0) for i, j in text_indices]
                    positive_pairs.extend([InputExample(texts=list(p[:2]), label=p[2]) for p in pos_pairs])

        # Create negative pairs efficiently
        all_texts = [(text, domain, concept) 
                    for domain in main_dict 
                    for concept in main_dict[domain] 
                    for text in main_dict[domain][concept]]
        
        # Use numpy for faster operations
        text_array = np.array(all_texts, dtype=object)
        indices = np.array(list(combinations(range(len(text_array)), 2)))
        
        for i, j in indices:
            text1, domain1, concept1 = text_array[i]
            text2, domain2, concept2 = text_array[j]
            if domain1 != domain2 or concept1 != concept2:
                negative_pairs.append(InputExample(texts=[text1, text2], label=0.0))

        # Balance dataset efficiently
        if len(negative_pairs) > len(positive_pairs):
            # Use numpy random for faster sampling
            neg_indices = np.random.choice(
                len(negative_pairs), 
                size=len(positive_pairs), 
                replace=False
            )
            negative_pairs = [negative_pairs[i] for i in neg_indices]
        
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        return all_pairs

[Previous code remains exactly the same until DataProcessor class...]

class EmbeddingTrainer:
    """Handles the training of the embedding model with optimizations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = SentenceTransformer(config.embedding_model)
        self.model.to(self.device)
        
    @torch.no_grad()
    def compute_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Compute embeddings efficiently in batches."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                device=self.device
            )
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)
        
    def train(self, train_pairs: List[InputExample], 
              val_pairs: List[InputExample], 
              output_path: str):
        """Train the embedding model with optimized batch processing."""
        logger.info("Starting embedding model training...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Optimize dataloader
        train_dataloader = DataLoader(
            train_pairs, 
            shuffle=True, 
            batch_size=self.config.batch_size,
            pin_memory=True if self.device.type == 'cuda' else False,
            num_workers=self.config.max_workers
        )
        
        # Initialize loss
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Create evaluator with optimized settings
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
            val_pairs, 
            batch_size=self.config.batch_size,
            name='validation',
            show_progress_bar=True
        )
        
        # Calculate warmup steps
        num_training_steps = len(train_dataloader) * self.config.num_epochs
        warmup_steps = int(num_training_steps * 0.1)
        
        # Train with optimized settings
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=self.config.num_epochs,
            evaluation_steps=500,
            warmup_steps=warmup_steps,
            output_path=output_path,
            save_best_model=True,
            show_progress_bar=True,
            use_amp=True if self.device.type == 'cuda' else False  # Automatic mixed precision
        )
        
        return self.model

class ClassificationTrainer:
    """Handles the training of the classification head with optimizations."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def train(self, embeddings: np.ndarray, 
              domains: np.ndarray, 
              concepts: np.ndarray, 
              output_path: str):
        """Train classification models with optimized parallel processing."""
        # Initialize classifiers with optimized settings
        if self.config.classification_head == 'logistic':
            domain_clf = LogisticRegression(
                max_iter=1000,
                n_jobs=-1,
                verbose=1,
                solver='saga'  # Faster for large datasets
            )
            concept_clf = LogisticRegression(
                max_iter=1000,
                n_jobs=-1,
                verbose=1,
                solver='saga'
            )
        else:  # xgboost
            domain_clf = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                n_jobs=-1,
                tree_method='gpu_hist' if self.config.device == 'cuda' else 'hist',
                predictor='gpu_predictor' if self.config.device == 'cuda' else 'cpu_predictor'
            )
            concept_clf = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                n_jobs=-1,
                tree_method='gpu_hist' if self.config.device == 'cuda' else 'hist',
                predictor='gpu_predictor' if self.config.device == 'cuda' else 'cpu_predictor'
            )
        
        # Train classifiers
        logger.info("Training domain classifier...")
        domain_clf.fit(embeddings, domains)
        
        logger.info("Training concept classifier...")
        concept_clf.fit(embeddings, concepts)
        
        # Save models efficiently
        joblib.dump(domain_clf, 
                   os.path.join(output_path, f'domain_classifier_{self.config.classification_head}.joblib'),
                   compress=3)
        joblib.dump(concept_clf, 
                   os.path.join(output_path, f'concept_classifier_{self.config.classification_head}.joblib'),
                   compress=3)
        
        return domain_clf, concept_clf

class Predictor:
    """Handles predictions with optimized batch processing."""
    
    def __init__(self, model_path: str, description_refine: bool = False):
        self.model_path = Path(model_path)
        self.description_refine = description_refine
        
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Load configuration
        with open(self.model_path / 'config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model with optimized settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model = SentenceTransformer(str(self.model_path / 'embedding_model'))
        self.embedding_model.to(self.device)
        
        # Load classifiers and encoders efficiently
        self._load_classifiers()
        self._load_encoders()
        
        # Initialize rate limiter for API calls
        self.rate_limiter = AdaptiveRateLimiter()
    
    def _load_classifiers(self):
        """Load classifiers efficiently."""
        classifier_type = self.config['classification_head']
        
        domain_clf_path = self.model_path / f'domain_classifier_{classifier_type}.joblib'
        concept_clf_path = self.model_path / f'concept_classifier_{classifier_type}.joblib'
        
        if not domain_clf_path.exists() or not concept_clf_path.exists():
            raise ValueError("Classifier files not found")
        
        # Load in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_domain = executor.submit(joblib.load, domain_clf_path)
            future_concept = executor.submit(joblib.load, concept_clf_path)
            
            self.domain_clf = future_domain.result()
            self.concept_clf = future_concept.result()
    
    def _load_encoders(self):
        """Load encoders efficiently."""
        encoders_path = self.model_path / 'label_encoders.json'
        if not encoders_path.exists():
            raise ValueError("Label encoders file not found")
            
        with open(encoders_path, 'r') as f:
            encoders = json.load(f)
            self.domain_classes = encoders['domain_classes']
            self.concept_classes = encoders['concept_classes']
    
    @lru_cache(maxsize=1024)
    def standardize_text(self, attr_name: str, description: str) -> str:
        """Standardize text with caching and rate limiting."""
        if not self.description_refine:
            return description
            
        try:
            self.rate_limiter.wait()
            response = chinou_response(f"""Please standardize this attribute description to match analytics metric pattern:
            Attribute: {attr_name}
            Description: {description}
            Output a single standardized description focused on measurement and purpose.""")
            self.rate_limiter.success()
            return response
        except Exception as e:
            logger.warning(f"Failed to refine description: {e}")
            self.rate_limiter.failure()
            return description
    
    @torch.no_grad()
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings efficiently using batching."""
        return self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            device=self.device
        )
    
    def predict_single(self, attr_name: str, description: str) -> Tuple[str, str]:
        """Predict for single text efficiently."""
        processed_text = self.standardize_text(attr_name, description)
        embedding = self._get_embeddings([processed_text])
        
        domain_pred = self.domain_clf.predict(embedding)[0]
        concept_pred = self.concept_clf.predict(embedding)[0]
        
        return (
            self.domain_classes[domain_pred],
            self.concept_classes[concept_pred]
        )
    
    def predict_batch(self, attr_names: List[str], descriptions: List[str]) -> List[Tuple[str, str]]:
        """Predict for batch of texts efficiently."""
        if len(attr_names) != len(descriptions):
            raise ValueError("Number of attribute names must match number of descriptions")
        
        # Process texts in parallel batches
        with ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4)) as executor:
            batch_size = 32
            futures = []
            
            for i in range(0, len(descriptions), batch_size):
                batch_attrs = attr_names[i:i+batch_size]
                batch_descs = descriptions[i:i+batch_size]
                
                future = executor.submit(
                    lambda x: [self.standardize_text(attr, desc) 
                             for attr, desc in zip(*x)],
                    (batch_attrs, batch_descs)
                )
                futures.append(future)
            
            processed_texts = []
            for future in tqdm(as_completed(futures), 
                             total=len(futures),
                             desc="Processing descriptions"):
                processed_texts.extend(future.result())
        
        # Get embeddings efficiently
        embeddings = self._get_embeddings(processed_texts)
        
        # Get predictions
        domain_preds = self.domain_clf.predict(embeddings)
        concept_preds = self.concept_clf.predict(embeddings)
        
        return [
            (self.domain_classes[d], self.concept_classes[c])
            for d, c in zip(domain_preds, concept_preds)
        ]

[Previous code remains the same until Predictor class...]

class ModelTracker:
    """Track and manage model experiments with optimized storage."""
    
    def __init__(self, base_path: str = 'experiments'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.history_file = self.base_path / 'experiment_history.json'
        self.history = []
        self._load_history()
        
    def _load_history(self):
        """Load experiment history efficiently."""
        if self.history_file.exists():
            try:
                # Read file in chunks for memory efficiency
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except json.JSONDecodeError:
                logger.error("Corrupted history file. Creating backup and starting fresh.")
                if self.history_file.exists():
                    backup_path = self.history_file.with_suffix('.json.bak')
                    self.history_file.rename(backup_path)
                self.history = []
    
    def save_experiment(self, 
                       experiment_path: str, 
                       config: Dict, 
                       metrics: Dict[str, float]):
        """Save experiment details with atomic write."""
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'path': experiment_path,
            'config': config,
            'metrics': metrics
        }
        
        self.history.append(experiment)
        
        # Atomic write to prevent corruption
        temp_file = self.history_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(self.history, f, indent=2)
            temp_file.replace(self.history_file)
        except Exception as e:
            logger.error(f"Error saving experiment history: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    def get_best_model(self, metric: str = 'domain_accuracy') -> str:
        """Get best model path efficiently."""
        if not self.history:
            raise ValueError("No experiments found")
        
        # Use max with key function instead of sorting
        best_experiment = max(
            self.history,
            key=lambda x: x['metrics'].get(metric, float('-inf'))
        )
        
        return best_experiment['path']

def batch_process_texts(texts: List[str], 
                       process_fn: callable, 
                       batch_size: int = 32, 
                       max_workers: int = 4) -> List:
    """Process texts in optimized parallel batches."""
    # Pre-allocate results list for better memory efficiency
    results = [None] * len(texts)
    
    # Create batches using numpy for efficiency
    indices = np.arange(0, len(texts), batch_size)
    batches = [(i, texts[i:i+batch_size]) for i in indices]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_index = {
            executor.submit(process_fn, batch): (start_idx, len(batch))
            for start_idx, batch in batches
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_index), 
                         total=len(future_to_index),
                         desc="Processing batches"):
            start_idx, batch_len = future_to_index[future]
            try:
                batch_results = future.result()
                results[start_idx:start_idx+batch_len] = batch_results
            except Exception as e:
                logger.error(f"Error processing batch starting at {start_idx}: {e}")
                raise
    
    return results

def train_pipeline(config: Config, 
                  train_data: pd.DataFrame,
                  base_path: str = 'experiments') -> str:
    """Optimized training pipeline."""
    try:
        # Setup experiment folder
        experiment_path = setup_folders(base_path, config.classification_head)
        logger.info(f"Starting experiment in {experiment_path}")
        
        # Save configuration atomically
        config_path = Path(experiment_path) / 'config.yaml'
        temp_config_path = config_path.with_suffix('.tmp')
        with open(temp_config_path, 'w') as f:
            yaml.dump(vars(config), f)
        temp_config_path.replace(config_path)
        
        # Initialize processors
        data_processor = DataProcessor(config)
        embedding_trainer = EmbeddingTrainer(config)
        classification_trainer = ClassificationTrainer(config)
        
        # Process descriptions with progress tracking
        with tqdm(total=4, desc="Training Pipeline Progress") as pbar:
            # Step 1: Process descriptions
            processed_df = data_processor.process_descriptions(train_data)
            pbar.update(1)
            
            # Step 2: Create and split pairs
            all_pairs = data_processor.create_pairs(processed_df)
            train_pairs, val_pairs = train_test_split(
                all_pairs, 
                test_size=config.test_size,
                random_state=config.random_state
            )
            pbar.update(1)
            
            # Step 3: Train embedding model
            embedding_model = embedding_trainer.train(
                train_pairs,
                val_pairs,
                os.path.join(experiment_path, 'embedding_model')
            )
            pbar.update(1)
            
            # Step 4: Generate embeddings and train classifiers
            texts = (processed_df['processed_description'] 
                    if 'processed_description' in processed_df.columns 
                    else processed_df['description'])
            
            # Generate embeddings efficiently
            embeddings = embedding_model.encode(
                texts.tolist(),
                batch_size=config.batch_size,
                show_progress_bar=True
            )
            
            # Encode labels
            domains = data_processor.domain_encoder.fit_transform(processed_df['domain'])
            concepts = data_processor.concept_encoder.fit_transform(processed_df['concept'])
            
            # Save encoders atomically
            encoders = {
                'domain_classes': data_processor.domain_encoder.classes_.tolist(),
                'concept_classes': data_processor.concept_encoder.classes_.tolist()
            }
            encoders_path = Path(experiment_path) / 'label_encoders.json'
            temp_encoders_path = encoders_path.with_suffix('.tmp')
            with open(temp_encoders_path, 'w') as f:
                json.dump(encoders, f, indent=2)
            temp_encoders_path.replace(encoders_path)
            
            # Train classifiers
            classification_trainer.train(
                embeddings,
                domains,
                concepts,
                experiment_path
            )
            pbar.update(1)
        
        logger.info(f"Training completed successfully. Models saved in {experiment_path}")
        return experiment_path
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}", exc_info=True)
        # Cleanup incomplete experiment
        if experiment_path:
            shutil.rmtree(experiment_path, ignore_errors=True)
        raise

def evaluate_model(predictor: Predictor, test_df: pd.DataFrame) -> Dict[str, float]:
    """Evaluate model with optimized batch processing."""
    try:
        if 'attribute_name' not in test_df.columns:
            test_df['attribute_name'] = test_df.index.astype(str)
        
        # Convert to numpy arrays for faster processing
        texts = test_df['description'].values
        attr_names = test_df['attribute_name'].values
        true_domains = test_df['domain'].values
        true_concepts = test_df['concept'].values
        
        # Get predictions efficiently
        predictions = predictor.predict_batch(attr_names.tolist(), texts.tolist())
        pred_domains, pred_concepts = zip(*predictions)
        
        # Use numpy for faster calculations
        domain_accuracy = np.mean(np.array(pred_domains) == true_domains)
        concept_accuracy = np.mean(np.array(pred_concepts) == true_concepts)
        
        # Calculate detailed metrics
        domain_metrics = classification_report(true_domains, pred_domains, output_dict=True)
        concept_metrics = classification_report(true_concepts, pred_concepts, output_dict=True)
        
        metrics = {
            'domain_accuracy': float(domain_accuracy),
            'concept_accuracy': float(concept_accuracy),
            'domain_metrics': domain_metrics,
            'concept_metrics': concept_metrics
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Hierarchical Text Classification')
        
        parser.add_argument('--mode', 
                          choices=['train', 'predict', 'evaluate'],
                          required=True,
                          help='Operation mode')
        parser.add_argument('--input_file',
                          required=True,
                          help='Path to input CSV file')
        parser.add_argument('--description_refine',
                          action='store_true',
                          help='Whether to refine descriptions using LLM')
        parser.add_argument('--classification_head',
                          choices=['logistic', 'xgboost'],
                          default='logistic',
                          help='Type of classification model')
        parser.add_argument('--model_path',
                          help='Path to saved model (for predict/evaluate mode)')
        parser.add_argument('--batch_size',
                          type=int,
                          default=32,
                          help='Batch size for processing')
        parser.add_argument('--num_workers',
                          type=int,
                          default=min(4, os.cpu_count() or 1),
                          help='Number of parallel workers')
        
        args = parser.parse_args()
        
        # Train mode
        if args.mode == 'train':
            config = Config(
                description_refine=args.description_refine,
                classification_head=args.classification_head,
                batch_size=args.batch_size,
                max_workers=args.num_workers
            )
            
            # Load and validate input data efficiently
            input_path = Path(args.input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {args.input_file}")
            
            # Read CSV in chunks for memory efficiency
            chunk_size = 10000
            chunks = []
            for chunk in pd.read_csv(input_path, chunksize=chunk_size):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            
            required_columns = {'description', 'domain', 'concept'}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Input CSV must contain columns: {required_columns}")
            
            # Split data efficiently
            train_idx, test_idx = train_test_split(
                np.arange(len(df)),
                test_size=0.2,
                random_state=42
            )
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            # Train and evaluate
            experiment_path = train_pipeline(config, train_df)
            predictor = Predictor(experiment_path)
            metrics = evaluate_model(predictor, test_df)
            
            # Track experiment
            tracker = ModelTracker()
            tracker.save_experiment(
                experiment_path,
                vars(config),
                metrics
            )
            
            # Print results
            print("\nTraining completed!")
            print(f"Models saved in: {experiment_path}")
            print("\nTest Set Metrics:")
            print(f"Domain Accuracy: {metrics['domain_accuracy']:.4f}")
            print(f"Concept Accuracy: {metrics['concept_accuracy']:.4f}")
        
        # Prediction mode
        elif args.mode == 'predict':
            if not args.model_path:
                tracker = ModelTracker()
                best_logistic = tracker.get_best_model('domain_accuracy')
                best_xgboost = tracker.get_best_model('concept_accuracy')
                
                print(f"\nAvailable models:")
                print(f"1. Best Logistic Regression model: {best_logistic}")
                print(f"2. Best XGBoost model: {best_xgboost}")
                
                model_choice = input("\nSelect model type (1/2): ")
                model_path = best_logistic if model_choice == "1" else best_xgboost
            else:
                model_path = args.model_path
            
            # Initialize predictor
            predictor = Predictor(model_path, description_refine=args.description_refine)
            
            input_path = Path(args.input_file)
            if input_path.exists():
                # Process CSV in chunks for memory efficiency
                output_chunks = []
                chunk_size = 10000
                
                for chunk in pd.read_csv(input_path, chunksize=chunk_size):
                    if 'attribute_name' not in chunk.columns:
                        chunk['attribute_name'] = chunk.index.astype(str)
                    
                    predictions = predictor.predict_batch(
                        chunk['attribute_name'].tolist(),
                        chunk['description'].tolist()
                    )
                    
                    chunk['predicted_domain'], chunk['predicted_concept'] = zip(*predictions)
                    output_chunks.append(chunk)
                
                # Combine results and save
                df = pd.concat(output_chunks, ignore_index=True)
                output_path = Path('predictions') / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                output_path.parent.mkdir(exist_ok=True)
                
                # Save in chunks for memory efficiency
                for i, chunk in enumerate(np.array_split(df, 10)):
                    mode = 'w' if i == 0 else 'a'
                    header = i == 0
                    chunk.to_csv(output_path, index=False, mode=mode, header=header)
                
                print(f"\nPredictions saved to: {output_path}")
            
            else:
                # Single text prediction
                try:
                    attr_name, description = args.input_file.split(':', 1)
                except ValueError:
                    attr_name = "unknown"
                    description = args.input_file
                
                domain, concept = predictor.predict_single(attr_name.strip(), description.strip())
                
                print(f"\nPrediction Results:")
                print(f"Attribute: {attr_name}")
                if args.description_refine:
                    print(f"Original Description: {description}")
                    print(f"Refined Description: {predictor.standardize_text(attr_name, description)}")
                print(f"Predicted Domain: {domain}")
                print(f"Predicted Concept: {concept}")
        
# Evaluate mode
        else:  
            if not args.model_path:
                tracker = ModelTracker()
                model_path = tracker.get_best_model()
                logger.info(f"Using best model from: {model_path}")
            else:
                model_path = args.model_path
                
            predictor = Predictor(model_path)
            
            # Validate and load test data efficiently
            input_path = Path(args.input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {args.input_file}")
            
            # Process test data in chunks
            chunks = []
            chunk_size = 10000
            required_columns = {'description', 'domain', 'concept'}
            
            for chunk in pd.read_csv(input_path, chunksize=chunk_size):
                if not required_columns.issubset(chunk.columns):
                    raise ValueError(f"Input CSV must contain columns: {required_columns}")
                chunks.append(chunk)
            
            test_df = pd.concat(chunks, ignore_index=True)
            
            # Evaluate in batches
            metrics = evaluate_model(predictor, test_df)
            
            # Print results efficiently
            print("\nEvaluation Results:")
            print(f"Domain Accuracy: {metrics['domain_accuracy']:.4f}")
            print(f"Concept Accuracy: {metrics['concept_accuracy']:.4f}")
            print("\nDetailed Metrics:")
            
            # Process predictions in chunks for memory efficiency
            all_domain_preds = []
            all_concept_preds = []
            
            for i in range(0, len(test_df), chunk_size):
                chunk = test_df.iloc[i:i+chunk_size]
                predictions = predictor.predict_batch(
                    chunk['attribute_name'].tolist() if 'attribute_name' in chunk.columns 
                    else chunk.index.astype(str).tolist(),
                    chunk['description'].tolist()
                )
                domain_preds, concept_preds = zip(*predictions)
                all_domain_preds.extend(domain_preds)
                all_concept_preds.extend(concept_preds)
            
            print("\nDomain Classification Report:")
            print(classification_report(test_df['domain'], all_domain_preds))
            print("\nConcept Classification Report:")
            print(classification_report(test_df['concept'], all_concept_preds))
            
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise
    finally:
        # Cleanup and resource handling
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Process completed")
