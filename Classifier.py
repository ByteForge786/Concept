import os
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report  # Added missing import
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
from test import *  # Import chinou_response from test module

# Configure logging with proper path handling
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

class DataProcessor:
    """Handles data processing and pair creation for training."""
    
    def __init__(self, config: Config):
        self.config = config
        self.domain_encoder = LabelEncoder()
        self.concept_encoder = LabelEncoder()
        
    def standardize_description_batch(self, batch: List[Tuple[str, str]]) -> List[str]:
        """Standardize a batch of descriptions using LLM."""
        standardized = []
        for attr_name, desc in batch:
            try:
                std_desc = chinou_response(f"""Please standardize this attribute description to match analytics metric pattern:
                Attribute: {attr_name}
                Description: {desc}
                Output a single standardized description focused on measurement and purpose.""")
                standardized.append(std_desc)
            except Exception as e:
                logger.error(f"Error standardizing description: {e}")
                standardized.append(desc)  # Fallback to original
        return standardized

    def process_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all descriptions in parallel batches."""
        if not self.config.description_refine:
            return df
            
        logger.info("Starting description standardization...")
        batch_size = 50
        batches = [
            list(zip(df['attribute_name'][i:i+batch_size], 
                    df['description'][i:i+batch_size]))
            for i in range(0, len(df), batch_size)
        ]
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(self.standardize_description_batch, batch) 
                      for batch in batches]
            
            standardized_descriptions = []
            for future in tqdm(as_completed(futures), total=len(futures)):
                standardized_descriptions.extend(future.result())
                
        df['processed_description'] = standardized_descriptions
        return df

    def create_pairs(self, df: pd.DataFrame) -> List[InputExample]:
        """Create training pairs for contrastive learning."""
        main_dict = {}
        for _, row in df.iterrows():
            domain = row['domain']
            concept = row['concept']
            text = row['processed_description'] if 'processed_description' in df.columns else row['description']
            
            if domain not in main_dict:
                main_dict[domain] = {}
            if concept not in main_dict[domain]:
                main_dict[domain][concept] = []
            main_dict[domain][concept].append(text)

        positive_pairs = []
        negative_pairs = []
        
        # Create positive pairs (same concept)
        for domain in main_dict:
            for concept in main_dict[domain]:
                texts = main_dict[domain][concept]
                for text1, text2 in combinations(texts, 2):
                    positive_pairs.append(InputExample(texts=[text1, text2], label=1.0))

        # Create negative pairs (different concepts)
        all_texts = [(text, domain, concept) 
                    for domain in main_dict 
                    for concept in main_dict[domain] 
                    for text in main_dict[domain][concept]]
                    
        for (text1, domain1, concept1), (text2, domain2, concept2) in combinations(all_texts, 2):
            if domain1 != domain2 or concept1 != concept2:
                negative_pairs.append(InputExample(texts=[text1, text2], label=0.0))

        # Balance dataset
        if len(negative_pairs) > len(positive_pairs):
            negative_pairs = random.sample(negative_pairs, len(positive_pairs))
        
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        return all_pairs

class EmbeddingTrainer:
    """Handles the training of the embedding model."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)
        
    def train(self, train_pairs: List[InputExample], 
              val_pairs: List[InputExample], 
              output_path: str):
        """Train the embedding model."""
        logger.info("Starting embedding model training...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        train_dataloader = DataLoader(
            train_pairs, 
            shuffle=True, 
            batch_size=self.config.batch_size
        )
        
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
            val_pairs, 
            name='validation'
        )
        
        warmup_steps = int(len(train_dataloader) * self.config.num_epochs * 0.1)
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=self.config.num_epochs,
            evaluation_steps=500,
            warmup_steps=warmup_steps,
            output_path=output_path,
            save_best_model=True,
            show_progress_bar=True
        )
        
        return self.model

class ClassificationTrainer:
    """Handles the training of the classification head."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def train(self, embeddings: np.ndarray, 
              domains: np.ndarray, 
              concepts: np.ndarray, 
              output_path: str):
        """Train classification models for domain and concept."""
        if self.config.classification_head == 'logistic':
            domain_clf = LogisticRegression(max_iter=1000, n_jobs=-1)
            concept_clf = LogisticRegression(max_iter=1000, n_jobs=-1)
        else:  # xgboost
            domain_clf = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                n_jobs=-1
            )
            concept_clf = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                n_jobs=-1
            )
            
        logger.info("Training domain classifier...")
        domain_clf.fit(embeddings, domains)
        
        logger.info("Training concept classifier...")
        concept_clf.fit(embeddings, concepts)
        
        # Save models with classifier type in filename
        joblib.dump(domain_clf, os.path.join(output_path, f'domain_classifier_{self.config.classification_head}.joblib'))
        joblib.dump(concept_clf, os.path.join(output_path, f'concept_classifier_{self.config.classification_head}.joblib'))
        
        return domain_clf, concept_clf

class Predictor:
    """Handles predictions using trained models."""
    
    def __init__(self, model_path: str, description_refine: bool = False):
        self.model_path = Path(model_path)
        self.description_refine = description_refine
        
        # Ensure model path exists
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
            
        self.embedding_model = SentenceTransformer(str(self.model_path / 'embedding_model'))
        
        # Load configuration
        config_path = self.model_path / 'config.yaml'
        if not config_path.exists():
            raise ValueError(f"Config file not found at {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load classifiers
        classifier_type = self.config['classification_head']
        domain_clf_path = self.model_path / f'domain_classifier_{classifier_type}.joblib'
        concept_clf_path = self.model_path / f'concept_classifier_{classifier_type}.joblib'
        
        if not domain_clf_path.exists() or not concept_clf_path.exists():
            raise ValueError("Classifier files not found")
            
        self.domain_clf = joblib.load(domain_clf_path)
        self.concept_clf = joblib.load(concept_clf_path)
        
        # Load label encoders
        encoders_path = self.model_path / 'label_encoders.json'
        if not encoders_path.exists():
            raise ValueError("Label encoders file not found")
            
        with open(encoders_path, 'r') as f:
            encoders = json.load(f)
            self.domain_classes = encoders['domain_classes']
            self.concept_classes = encoders['concept_classes']
            
    def standardize_text(self, attr_name: str, description: str) -> str:
        """Standardize text description using LLM if enabled."""
        if not self.description_refine:
            return description
            
        try:
            return chinou_response(f"""Please standardize this attribute description to match analytics metric pattern:
            Attribute: {attr_name}
            Description: {description}
            Output a single standardized description focused on measurement and purpose.""")
        except Exception as e:
            logger.warning(f"Failed to refine description: {e}")
            return description
            
    def predict_single(self, attr_name: str, description: str) -> Tuple[str, str]:
        """Predict domain and concept for a single text."""
        processed_text = self.standardize_text(attr_name, description)
        embedding = self.embedding_model.encode([processed_text])
        
        domain_pred = self.domain_clf.predict(embedding)[0]
        concept_pred = self.concept_clf.predict(embedding)[0]
        
        return (
            self.domain_classes[domain_pred],
            self.concept_classes[concept_pred]
        )
        
    def predict_batch(self, attr_names: List[str], descriptions: List[str]) -> List[Tuple[str, str]]:
        """Predict domain and concept for a batch of texts."""
        if len(attr_names) != len(descriptions):
            raise ValueError("Number of attribute names must match number of descriptions")
            
        processed_texts = [
            self.standardize_text(attr, desc)
            for attr, desc in zip(attr_names, descriptions)
        ] if self.description_refine else descriptions
        
        embeddings = self.embedding_model.encode(processed_texts, batch_size=32, show_progress_bar=True)
        
        domain_preds = self.domain_clf.predict(embeddings)
        concept_preds = self.concept_clf.predict(embeddings)
        
        return [
            (self.domain_classes[d], self.concept_classes[c])
            for d, c in zip(domain_preds, concept_preds)
        ]

def setup_folders(base_path: str, classifier_type: str) -> str:
    """Create necessary folders for experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = Path(base_path) / classifier_type / timestamp
    experiment_path.mkdir(parents=True, exist_ok=True)
    return str(experiment_path)

def train_pipeline(config: Config, 
                  train_data: pd.DataFrame,
                  base_path: str = 'experiments') -> str:
    """Complete training pipeline."""
    try:
        # Setup experiment folder
        experiment_path = setup_folders(base_path, config.classification_head)
        logger.info(f"Starting experiment in {experiment_path}")
        
        # Save configuration
        with open(Path(experiment_path) / 'config.yaml', 'w') as f:
            yaml.dump(vars(config), f)
            
        # Initialize processors
        data_processor = DataProcessor(config)
        embedding_trainer = EmbeddingTrainer(config)
        classification_trainer = ClassificationTrainer(config)
        
        # Process descriptions
        processed_df = data_processor.process_descriptions(train_data)
        
        # Create pairs for contrastive learning
        all_pairs = data_processor.create_pairs(processed_df)
        train_pairs, val_pairs = train_test_split(
            all_pairs, 
            test_size=config.test_size,
            random_state=config.random_state
        )
        
        # Train embedding model
        embedding_model = embedding_trainer.train(
            train_pairs,
            val_pairs,
            os.path.join(experiment_path, 'embedding_model')
        )
        
        # Generate embeddings for classification
        texts = processed_df['processed_description'] if 'processed_description' in processed_df.columns else processed_df['description']
        embeddings = embedding_model.encode(texts.tolist(), batch_size=32, show_progress_bar=True)
        
        # Encode labels
        domains = data_processor.domain_encoder.fit_transform(processed_df['domain'])
        concepts = data_processor.concept_encoder.fit_transform(processed_df['concept'])
        
        # Save label encoders
        encoders = {
            'domain_classes': data_processor.domain_encoder.classes_.tolist(),
            'concept_classes': data_processor.concept_encoder.classes_.tolist()
        }
        with open(Path(experiment_path) / 'label_encoders.json', 'w') as f:
            json.dump(encoders, f, indent=2)
        
        # Train classifiers
        classification_trainer.train(
            embeddings,
            domains,
            concepts,
            experiment_path
        )
        
        logger.info(f"Training completed successfully. Models saved in {experiment_path}")
        return experiment_path
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}", exc_info=True)
        raise

def evaluate_model(predictor: Predictor, test_df: pd.DataFrame) -> Dict[str, float]:
    """Evaluate model performance on test set."""
    try:
        if 'attribute_name' not in test_df.columns:
            test_df['attribute_name'] = test_df.index.astype(str)
            
        texts = test_df['description'].tolist()
        attr_names = test_df['attribute_name'].tolist()
        true_domains = test_df['domain'].tolist()
        true_concepts = test_df['concept'].tolist()
        
        predictions = predictor.predict_batch(attr_names, texts)
        pred_domains, pred_concepts = zip(*predictions)
        
        # Calculate accuracy
        domain_accuracy = sum(p == t for p, t in zip(pred_domains, true_domains)) / len(true_domains)
        concept_accuracy = sum(p == t for p, t in zip(pred_concepts, true_concepts)) / len(true_concepts)
        
        # Calculate per-class metrics
        domain_metrics = classification_report(true_domains, pred_domains, output_dict=True)
        concept_metrics = classification_report(true_concepts, pred_concepts, output_dict=True)
        
        metrics = {
            'domain_accuracy': domain_accuracy,
            'concept_accuracy': concept_accuracy,
            'domain_metrics': domain_metrics,
            'concept_metrics': concept_metrics
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}", exc_info=True)
        raise

class ModelTracker:
    """Track and manage model experiments."""
    
    def __init__(self, base_path: str = 'experiments'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.history_file = self.base_path / 'experiment_history.json'
        self._load_history()
        
    def _load_history(self):
        """Load experiment history from file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []
            
    def save_experiment(self, 
                       experiment_path: str, 
                       config: Dict, 
                       metrics: Dict[str, float]):
        """Save experiment details and metrics."""
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'path': experiment_path,
            'config': config,
            'metrics': metrics
        }
        
        self.history.append(experiment)
        
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
            
    def get_best_model(self, metric: str = 'domain_accuracy') -> str:
        """Get path of best performing model based on specified metric."""
        if not self.history:
            raise ValueError("No experiments found")
            
        best_experiment = max(
            self.history,
            key=lambda x: x['metrics'][metric]
        )
        
        return best_experiment['path']

def batch_process_texts(texts: List[str], 
                       process_fn: callable, 
                       batch_size: int = 32, 
                       max_workers: int = 4) -> List:
    """Process texts in parallel batches."""
    batches = [texts[i:i + batch_size] 
              for i in range(0, len(texts), batch_size)]
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_fn, batch) 
                  for batch in batches]
        
        for future in tqdm(as_completed(futures), 
                          total=len(futures),
                          desc="Processing batches"):
            results.extend(future.result())
            
    return results

if __name__ == "__main__":
    try:
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
                          default=4,
                          help='Number of parallel workers')
                          
        args = parser.parse_args()
        
        if args.mode == 'train':
            # Training mode
            config = Config(
                description_refine=args.description_refine,
                classification_head=args.classification_head,
                batch_size=args.batch_size,
                max_workers=args.num_workers
            )
            
            input_path = Path(args.input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {args.input_file}")
                
            df = pd.read_csv(input_path)
            required_columns = {'description', 'domain', 'concept'}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Input CSV must contain columns: {required_columns}")
            
            # Split into train/test
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )
            
            # Train models
            experiment_path = train_pipeline(config, train_df)
            
            # Evaluate on test set
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
            
        elif args.mode == 'predict':
            # Prediction mode
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
                # Batch prediction for CSV
                df = pd.read_csv(input_path)
                if 'attribute_name' not in df.columns:
                    df['attribute_name'] = df.index.astype(str)
                
                if 'description' not in df.columns:
                    raise ValueError("Input CSV must contain 'description' column")
                
                predictions = predictor.predict_batch(
                    df['attribute_name'].tolist(),
                    df['description'].tolist()
                )
                
                df['predicted_domain'], df['predicted_concept'] = zip(*predictions)
                
                # Save predictions
                output_path = Path('predictions') / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                output_path.parent.mkdir(exist_ok=True)
                df.to_csv(output_path, index=False)
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
                
        else:  # evaluate mode
            if not args.model_path:
                tracker = ModelTracker()
                model_path = tracker.get_best_model()
                logger.info(f"Using best model from: {model_path}")
            else:
                model_path = args.model_path
                
            predictor = Predictor(model_path)
            
            input_path = Path(args.input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {args.input_file}")
                
            test_df = pd.read_csv(input_path)
            required_columns = {'description', 'domain', 'concept'}
            if not required_columns.issubset(test_df.columns):
                raise ValueError(f"Input CSV must contain columns: {required_columns}")
            
            metrics = evaluate_model(predictor, test_df)
            
            print("\nEvaluation Results:")
            print(f"Domain Accuracy: {metrics['domain_accuracy']:.4f}")
            print(f"Concept Accuracy: {metrics['concept_accuracy']:.4f}")
            print("\nDetailed Metrics:")
            print("\nDomain Classification Report:")
            print(classification_report(
                test_df['domain'],
                [p[0] for p in predictor.predict_batch(
                    test_df['attribute_name'].tolist() if 'attribute_name' in test_df.columns 
                    else test_df.index.astype(str).tolist(),
                    test_df['description'].tolist()
                )]
            ))
            print("\nConcept Classification Report:")
            print(classification_report(
                test_df['concept'],
                [p[1] for p in predictor.predict_batch(
                    test_df['attribute_name'].tolist() if 'attribute_name' in test_df.columns 
                    else test_df.index.astype(str).tolist(),
                    test_df['description'].tolist()
                )]
            ))
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise
