import os
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers.losses import InfoNCE
import json
from tqdm import tqdm
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging with both file and console output
def setup_logging(log_dir: str = "logs"):
    """Setup detailed logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters and handlers
    file_handler = RotatingFileHandler(
        f"{log_dir}/attribute_classifier.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

@dataclass
class Config:
    """Configuration for the attribute classifier"""
    # Model settings
    model_name: str = 'all-MiniLM-L6-v2'
    max_seq_length: int = 128
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    
    # Training settings
    num_positives: int = 4  # Number of positive pairs per anchor
    num_negatives: int = 8  # Number of negative pairs per anchor
    temperature: float = 0.07
    validation_split: float = 0.1
    
    # Performance settings
    num_workers: int = 4
    max_threads: int = 8
    
    # Paths
    model_save_path: str = 'models/attribute-classifier'
    cache_dir: str = 'cache'
    log_dir: str = 'logs'
    metrics_dir: str = 'metrics'
    
    # Template
    template: str = "In the domain of {domain}, this attribute represents {concept} which is defined as {definition}"

    def __post_init__(self):
        """Create necessary directories"""
        for directory in [self.model_save_path, self.cache_dir, 
                         self.log_dir, self.metrics_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

class DataStats:
    """Track and log data statistics"""
    def __init__(self, config: Config):
        self.config = config
        self.stats = defaultdict(dict)
        
    def log_data_distribution(self, 
                            attributes_df: pd.DataFrame,
                            phase: str = "initial"):
        """Log distribution of data across domains and concepts"""
        logger.info(f"Data distribution - {phase}")
        
        # Domain-Concept distribution
        distribution = attributes_df.groupby(['domain', 'concept']).size()
        for (domain, concept), count in distribution.items():
            logger.info(f"Domain: {domain}, Concept: {concept}, Count: {count}")
            self.stats[phase][f"{domain}-{concept}"] = count
            
        # Save distribution plot
        plt.figure(figsize=(12, 8))
        distribution.plot(kind='bar')
        plt.title(f'Data Distribution - {phase}')
        plt.tight_layout()
        plt.savefig(f"{self.config.metrics_dir}/distribution_{phase}.png")
        plt.close()
        
    def save_stats(self):
        """Save statistics to file"""
        with open(f"{self.config.metrics_dir}/data_stats.json", 'w') as f:
            json.dump(self.stats, f, indent=4)

class AttributeDataProcessor:
    """Process and prepare data for training"""
    def __init__(self, config: Config):
        self.config = config
        self.stats = DataStats(config)
        
    def validate_data(self, 
                     attributes_df: pd.DataFrame, 
                     templates_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Validate input data format and content"""
        logger.info("Validating input data...")
        
        # Check required columns
        attr_required = ['attribute_name', 'description', 'domain', 'concept']
        tmpl_required = ['domain_name', 'concept_name', 'concept_definition']
        
        for col in attr_required:
            if col not in attributes_df.columns:
                raise ValueError(f"Missing required column in attributes CSV: {col}")
                
        for col in tmpl_required:
            if col not in templates_df.columns:
                raise ValueError(f"Missing required column in templates CSV: {col}")
        
        # Remove any duplicates
        attributes_df = attributes_df.drop_duplicates()
        templates_df = templates_df.drop_duplicates()
        
        # Log initial stats
        self.stats.log_data_distribution(attributes_df, "initial")
        
        return attributes_df, templates_df
        
    def prepare_data(self, 
                    attributes_df: pd.DataFrame, 
                    templates_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Process raw data and create necessary mappings"""
        logger.info("Preparing data mappings...")
        
        # Validate data
        attributes_df, templates_df = self.validate_data(attributes_df, templates_df)
        
        # Create domain-concept pairs
        attributes_df['label'] = attributes_df['domain'] + '-' + attributes_df['concept']
        
        # Create template mappings
        template_mappings = {}
        for _, row in templates_df.iterrows():
            key = f"{row['domain_name']}-{row['concept_name']}"
            template_mappings[key] = self.config.template.format(
                domain=row['domain_name'],
                concept=row['concept_name'],
                definition=row['concept_definition']
            )
            logger.debug(f"Created template for {key}: {template_mappings[key]}")
            
        # Create attribute mappings
        attribute_mappings = defaultdict(list)
        for _, row in attributes_df.iterrows():
            label = row['label']
            text = f"{row['attribute_name']} {row['description']}"
            attribute_mappings[label].append(text)
            
        logger.info(f"Created mappings for {len(template_mappings)} templates and {len(attribute_mappings)} labels")
        
        return dict(template_mappings), dict(attribute_mappings)

class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with InfoNCE loss"""
    def __init__(self, 
                 attribute_mappings: Dict[str, List[str]], 
                 template_mappings: Dict[str, str],
                 config: Config,
                 phase: str = "train"):
        self.attribute_mappings = attribute_mappings
        self.template_mappings = template_mappings
        self.config = config
        self.phase = phase
        self.labels = list(attribute_mappings.keys())
        
        logger.info(f"Initialized {phase} dataset with {len(self.labels)} labels")
        self._log_dataset_stats()
        
    def _log_dataset_stats(self):
        """Log dataset statistics"""
        stats = {
            "total_labels": len(self.labels),
            "total_attributes": sum(len(attrs) for attrs in self.attribute_mappings.values()),
            "attributes_per_label": {
                label: len(attrs) for label, attrs in self.attribute_mappings.items()
            }
        }
        logger.info(f"{self.phase} dataset stats: {json.dumps(stats, indent=2)}")
        
    def __len__(self):
        return sum(len(attrs) for attrs in self.attribute_mappings.values())
    
    def get_training_triplet(self, 
                           idx: int) -> Tuple[str, List[str], List[str]]:
        """Get anchor, positive and negative examples"""
        # Select random label and anchor
        anchor_label = random.choice(self.labels)
        anchor_text = random.choice(self.attribute_mappings[anchor_label])
        anchor_template = self.template_mappings[anchor_label]
        
        # Get positive examples
        positive_texts = [
            text for text in self.attribute_mappings[anchor_label]
            if text != anchor_text
        ]
        if len(positive_texts) < self.config.num_positives:
            positive_texts = positive_texts * (self.config.num_positives // len(positive_texts) + 1)
        positive_texts = random.sample(positive_texts, self.config.num_positives)
        
        # Get negative examples
        other_labels = [l for l in self.labels if l != anchor_label]
        negative_texts = []
        for _ in range(self.config.num_negatives):
            neg_label = random.choice(other_labels)
            neg_text = random.choice(self.attribute_mappings[neg_label])
            negative_texts.append(neg_text)
            
        return anchor_text, anchor_template, positive_texts, negative_texts
    
    def __getitem__(self, idx):
        anchor_text, anchor_template, positive_texts, negative_texts = self.get_training_triplet(idx)
        
        # Combine all texts for InfoNCE loss
        texts = [anchor_text, anchor_template] + positive_texts + negative_texts
        
        return {'texts': texts}

class MetricsTracker:
    """Track and log training metrics"""
    def __init__(self, config: Config):
        self.config = config
        self.metrics = defaultdict(list)
        
    def update(self, phase: str, metrics: Dict):
        """Update metrics for a phase"""
        for k, v in metrics.items():
            self.metrics[f"{phase}_{k}"].append(v)
            
    def log_metrics(self, epoch: int):
        """Log current metrics"""
        for k, v in self.metrics.items():
            if v:  # If there are values for this metric
                logger.info(f"Epoch {epoch} - {k}: {v[-1]:.4f}")
                
    def save_metrics(self):
        """Save metrics to file"""
        metrics_path = Path(self.config.metrics_dir) / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
        # Plot metrics
        self._plot_metrics()
        
    def _plot_metrics(self):
        """Plot training metrics"""
        plt.figure(figsize=(12, 8))
        for metric_name, values in self.metrics.items():
            plt.plot(values, label=metric_name)
        plt.legend()
        plt.title('Training Metrics')
        plt.tight_layout()
        plt.savefig(f"{self.config.metrics_dir}/training_metrics.png")
        plt.close()

class AttributeClassifier:
    """Main classifier using contrastive learning with InfoNCE loss"""
    def __init__(self, config: Config):
        self.config = config
        self.model = SentenceTransformer(config.model_name)
        self.metrics = MetricsTracker(config)
        
    def train(self, 
             attributes_df: pd.DataFrame, 
             templates_df: pd.DataFrame):
        """Train the model"""
        logger.info("Starting training pipeline...")
        
        # Prepare data
        processor = AttributeDataProcessor(self.config)
        template_mappings, attribute_mappings = processor.prepare_data(
            attributes_df, templates_df
        )
        
        # Split data for validation
        train_mappings, val_mappings = self._split_data(attribute_mappings)
        
        # Create datasets
        train_dataset = ContrastiveDataset(
            train_mappings, template_mappings, self.config, "train"
        )
        val_dataset = ContrastiveDataset(
            val_mappings, template_mappings, self.config, "val"
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        
        # Initialize loss
        train_loss = InfoNCE(
            model=self.model,
            temperature=self.config.temperature
        )
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{self.config.num_epochs}")
            
            # Training phase
            self.model.train()
            train_metrics = self._train_epoch(train_dataloader, train_loss)
            self.metrics.update("train", train_metrics)
            
            # Validation phase
            self.model.eval()
            val_metrics = self._validate_epoch(val_dataloader, train_loss)
            self.metrics.update("val", val_metrics)
            
            # Log metrics
            self.metrics.log_metrics(epoch)
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model('best')
                
        # Save final metrics and model
        self.metrics.save_metrics()
        self.save_model('final')
        
        logger.info("Training completed successfully")
        
    def _split_data(self, 
                    attribute_mappings: Dict[str, List[str]]) -> Tuple[Dict, Dict]:
        """Split data into train and validation sets"""
        train_mappings = {}
        val_mappings = {}
        
        for label, texts in attribute_mappings.items():
            if len(texts) > 1:  # Only split if we have enough samples
                train_texts, val_texts = train_test_split(
                    texts,
                    test_size=self.config.validation_split,
                    random_state=42
                )
                train_mappings[label] = train_texts
                val_mappings[label] = val_texts
            else:
                train_mappings[label] = texts
                val_mappings[label] = texts
                
        return train_mappings, val_mappings
        
    def _train_epoch(self, 
                    dataloader: DataLoader,
                    loss_fn: InfoNCE) -> Dict:
        """Train for one epoch"""
        total_loss = 0
        num_batches = len(dataloader)
        
        with tqdm(dataloader, desc="Training") as pbar:
            for batch in pbar:
                loss = loss_fn(batch)
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        return {'loss': total_loss / num_batches}
        
def _validate_epoch(self, 
                       dataloader: DataLoader,
                       loss_fn: InfoNCE) -> Dict:
        """Validate for one epoch"""
        total_loss = 0
        num_batches = len(dataloader)
        
        with torch.no_grad(), tqdm(dataloader, desc="Validation") as pbar:
            for batch in pbar:
                loss = loss_fn(batch)
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        return {'loss': total_loss / num_batches}
    
    def save_model(self, suffix: str):
        """Save model with given suffix"""
        save_path = f"{self.config.model_save_path}/model_{suffix}"
        self.model.save(save_path)
        logger.info(f"Saved model to {save_path}")
        
        # Save template mappings
        if hasattr(self, 'template_mappings'):
            with open(f"{save_path}/template_mappings.json", 'w') as f:
                json.dump(self.template_mappings, f, indent=4)
    
    def load_model(self, suffix: str):
        """Load model with given suffix"""
        load_path = f"{self.config.model_save_path}/model_{suffix}"
        self.model = SentenceTransformer(load_path)
        
        # Load template mappings
        template_path = f"{load_path}/template_mappings.json"
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                self.template_mappings = json.load(f)
        
        logger.info(f"Loaded model from {load_path}")
    
    def predict_single(self, 
                      text: str) -> Dict[str, Union[str, float]]:
        """Predict for a single text input"""
        logger.info(f"Predicting for text: {text}")
        
        if not hasattr(self, 'template_mappings'):
            raise ValueError("Model not trained or templates not loaded")
            
        # Get embeddings
        text_embedding = self.model.encode([text])[0]
        template_texts = list(self.template_mappings.values())
        template_embeddings = self.model.encode(template_texts)
        
        # Calculate similarities
        similarities = torch.nn.functional.cosine_similarity(
            torch.tensor(text_embedding).unsqueeze(0),
            torch.tensor(template_embeddings)
        )
        
        # Get top predictions
        top_indices = similarities.argsort(descending=True)[:3]
        template_labels = list(self.template_mappings.keys())
        
        predictions = []
        for idx in top_indices:
            label = template_labels[idx]
            domain, concept = label.split('-')
            predictions.append({
                'domain': domain,
                'concept': concept,
                'confidence': similarities[idx].item()
            })
            
        logger.info(f"Predictions for '{text}': {predictions[0]}")
        return predictions
    
    def predict_batch(self, 
                     texts: List[str],
                     batch_size: Optional[int] = None) -> List[Dict[str, Union[str, float]]]:
        """Predict for a batch of texts"""
        logger.info(f"Predicting for batch of {len(texts)} texts")
        
        if batch_size is None:
            batch_size = self.config.batch_size
            
        all_predictions = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Batch prediction"):
            batch_texts = texts[i:i+batch_size]
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
                batch_predictions = list(executor.map(self.predict_single, batch_texts))
            
            all_predictions.extend(batch_predictions)
            
        return all_predictions
    
    def predict_csv(self,
                   input_path: str,
                   output_path: str,
                   text_col: str,
                   batch_size: Optional[int] = None) -> pd.DataFrame:
        """Predict for a CSV file"""
        logger.info(f"Processing CSV file: {input_path}")
        
        # Read CSV
        df = pd.read_csv(input_path)
        if text_col not in df.columns:
            raise ValueError(f"Column {text_col} not found in CSV")
            
        # Get predictions
        texts = df[text_col].tolist()
        predictions = self.predict_batch(texts, batch_size)
        
        # Add predictions to dataframe
        df['predicted_domain'] = [p[0]['domain'] for p in predictions]
        df['predicted_concept'] = [p[0]['concept'] for p in predictions]
        df['confidence'] = [p[0]['confidence'] for p in predictions]
        
        # Save results
        df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        
        return df

def main():
    """Example usage of the attribute classifier"""
    # Initialize config
    config = Config()
    
    # Load data
    try:
        attributes_df = pd.read_csv("data/attributes.csv")
        templates_df = pd.read_csv("data/templates.csv")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return
        
    # Initialize and train classifier
    classifier = AttributeClassifier(config)
    
    try:
        # Train
        classifier.train(attributes_df, templates_df)
        
        # Example predictions
        # Single prediction
        text = "customer_email Email address of the customer"
        prediction = classifier.predict_single(text)
        logger.info(f"Single prediction: {prediction}")
        
        # Batch prediction
        texts = [
            "customer_email Email address of the customer",
            "product_price Price of the product",
            "order_date Date of the order"
        ]
        predictions = classifier.predict_batch(texts)
        logger.info(f"Batch predictions: {predictions}")
        
        # CSV prediction
        classifier.predict_csv(
            "data/test.csv",
            "data/predictions.csv",
            "attribute_column"
        )
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main(),
