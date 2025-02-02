import concurrent.futures
from typing import List, Tuple
import pandas as pd
import time
import logging
from tqdm import tqdm

class DescriptionProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def standardize_single_description(self, attr_name: str, desc: str) -> str:
        """Standardize a single description using LLM."""
        prompt = f"""Please standardize this attribute description into a clear, concise format.
        
        Attribute: {attr_name}
        Description: {desc}
        
        Return in following format only:
        Standardized Description: [your standardized description here]
        """
        
        try:
            # Simple retry logic
            for attempt in range(3):
                try:
                    rate_limiter.wait()
                    response = chinou_response(prompt)
                    rate_limiter.success()
                    
                    # Just take the description after the prefix
                    if "Standardized Description:" in response:
                        return response.split("Standardized Description:")[1].strip()
                    else:
                        return response
                        
                except Exception as e:
                    rate_limiter.failure()
                    if attempt < 2:  # Don't wait on last attempt
                        time.sleep((2 ** attempt) * 1)  # 1, 2, 4 seconds
                    if attempt == 2:
                        self.logger.error(f"Failed to standardize after 3 attempts: {e}")
                        return desc
                    
        except Exception as e:
            self.logger.error(f"Error standardizing description: {e}")
            return desc

    def standardize_description_batch(self, batch: List[Tuple[str, str]], max_workers: int = None) -> List[str]:
        """Parallelized batch description standardization using threads."""
        # Use ThreadPoolExecutor for I/O-bound tasks like API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create futures for each description in the batch
            futures = [
                executor.submit(self.standardize_single_description, attr_name, desc)
                for attr_name, desc in batch
            ]
            
            # Collect results as they complete
            standardized = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    standardized.append(future.result())
                except Exception as e:
                    self.logger.error(f"Exception in thread: {e}")
                    standardized.append(desc)  # Fallback to original description
        
        return standardized

    def process_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all descriptions with parallelization."""
        if not self.config.description_refine:
            return df
        
        self.logger.info("Starting description standardization...")
        
        # Create batches
        batch_size = 5  # Small batch size for API calls
        batches = [
            list(zip(df['attribute_name'][i:i+batch_size], 
                    df['description'][i:i+batch_size]))
            for i in range(0, len(df), batch_size)
        ]
        
        # Process batches with progress bar and threading
        standardized_descriptions = []
        with tqdm(total=len(batches), desc="Standardizing descriptions") as pbar:
            # Limit concurrent threads to prevent overwhelming the API
            max_workers = min(10, len(batches))  # Adjust based on your API's rate limits
            
            # Use ThreadPoolExecutor for batch processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit batches to executor
                future_to_batch = {
                    executor.submit(self.standardize_description_batch, batch): batch 
                    for batch in batches
                }
                
                # Process completed batches
                for future in concurrent.futures.as_completed(future_to_batch):
                    try:
                        batch_results = future.result()
                        standardized_descriptions.extend(batch_results)
                        pbar.update(1)
                    except Exception as e:
                        self.logger.error(f"Batch processing error: {e}")
        
        df['processed_description'] = standardized_descriptions
        
        # Log some examples
        self.logger.info("Standardization examples:")
        for i in range(min(3, len(df))):
            self.logger.info(f"\nAttribute: {df['attribute_name'].iloc[i]}")
            self.logger.info(f"Original: {df['description'].iloc[i]}")
            self.logger.info(f"Standardized: {df['processed_description'].iloc[i]}")
        
        return df



def process_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
    # ... existing code ...
    
    df['processed_description'] = standardized_descriptions
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"standardized_descriptions_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved standardized descriptions to: {output_path}")
    
    return df


def standardize_description_batch(self, batch: List[Tuple[str, str]]) -> List[str]:
    """Standardize a batch of descriptions using LLM."""
    standardized = []
    
    for attr_name, desc in batch:
        prompt = f"""Please standardize this attribute description into a clear, concise format.
        
        Attribute: {attr_name}
        Description: {desc}
        
        Return in following format only:
        Standardized Description: [your standardized description here]
        """
        
        try:
            # Simple retry logic
            for attempt in range(3):
                try:
                    rate_limiter.wait()
                    response = chinou_response(prompt)
                    rate_limiter.success()
                    
                    # Just take the description after the prefix
                    if "Standardized Description:" in response:
                        std_desc = response.split("Standardized Description:")[1].strip()
                        standardized.append(std_desc)
                        break
                    else:
                        standardized.append(response)
                        break
                        
                except Exception as e:
                    rate_limiter.failure()
                    if attempt < 2:  # Don't wait on last attempt
                        time.sleep((2 ** attempt) * 1)  # 1, 2, 4 seconds
                    if attempt == 2:
                        standardized.append(desc)  # Use original on final failure
                        logger.error(f"Failed to standardize after 3 attempts: {e}")
                    
        except Exception as e:
            logger.error(f"Error standardizing description: {e}")
            standardized.append(desc)  # Fallback to original
    
    return standardized

def process_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
    """Process all descriptions."""
    if not self.config.description_refine:
        return df
        
    logger.info("Starting description standardization...")
    
    # Create batches
    batch_size = 5  # Small batch size for API calls
    batches = [
        list(zip(df['attribute_name'][i:i+batch_size], 
                df['description'][i:i+batch_size]))
        for i in range(0, len(df), batch_size)
    ]
    
    # Process batches with progress bar
    standardized_descriptions = []
    with tqdm(total=len(batches), desc="Standardizing descriptions") as pbar:
        for batch in batches:
            batch_results = self.standardize_description_batch(batch)
            standardized_descriptions.extend(batch_results)
            pbar.update(1)
    
    df['processed_description'] = standardized_descriptions
    
    # Log some examples
    logger.info("Standardization examples:")
    for i in range(min(3, len(df))):
        logger.info(f"\nAttribute: {df['attribute_name'].iloc[i]}")
        logger.info(f"Original: {df['description'].iloc[i]}")
        logger.info(f"Standardized: {df['processed_description'].iloc[i]}")
    
    return df




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
    experiment_path = None  # Initialize it here
    try:
        # Setup experiment folder
        experiment_path = setup_folders(base_path, config.classification_head)
        logger.info(f"Starting experiment in {experiment_path}")
        
        # Rest of your code...
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}", exc_info=True)
        # Cleanup incomplete experiment
        if experiment_path:  # Now this check is valid
            shutil.rmtree(experiment_path, ignore_errors=True)
        raise




def create_pairs(self, df: pd.DataFrame) -> List[InputExample]:
    """Create training pairs focusing on similarity-based negative pairs."""
    main_dict = {}
    # Pre-compute all descriptions for batch encoding
    all_descriptions = []
    desc_to_concept = {}  # Keep track of which concept each description belongs to
    
    # First pass: organize data and collect descriptions
    for _, row in df.iterrows():
        domain = row['domain']
        concept = row['concept']
        text = row['processed_description'] if 'processed_description' in df.columns else row['description']
        
        if domain not in main_dict:
            main_dict[domain] = {}
        if concept not in main_dict[domain]:
            main_dict[domain][concept] = []
            
        main_dict[domain][concept].append(text)
        all_descriptions.append(text)
        desc_to_concept[text] = concept

    # Initialize model once
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # Get unique concepts and encode them in one batch
    all_concepts = list(set(concept for domain in main_dict for concept in main_dict[domain]))
    concept_embeddings = model.encode(all_concepts, batch_size=32, show_progress_bar=False)
    concept_embeddings = concept_embeddings / np.linalg.norm(concept_embeddings, axis=1)[:, np.newaxis]
    
    # Create concept similarity matrix
    concept_similarities = {}
    for i, j in combinations(range(len(all_concepts)), 2):
        sim = float(np.dot(concept_embeddings[i], concept_embeddings[j]))
        if sim >= 0.7:  # Only store high similarities for efficiency
            concept_similarities[(all_concepts[i], all_concepts[j])] = sim
            concept_similarities[(all_concepts[j], all_concepts[i])] = sim
    
    # Encode all descriptions in one batch
    desc_embeddings = model.encode(all_descriptions, batch_size=32, show_progress_bar=False)
    desc_embeddings = desc_embeddings / np.linalg.norm(desc_embeddings, axis=1)[:, np.newaxis]
    
    positive_pairs = []
    concept_similar_negatives = []
    desc_similar_negatives = []
    
    # Create positive pairs
    for domain in main_dict:
        for concept in main_dict[domain]:
            texts = main_dict[domain][concept]
            for text1, text2 in combinations(texts, 2):
                positive_pairs.append(InputExample(texts=[text1, text2], label=1.0))
    
    # Find similar description pairs efficiently
    desc_similarities = np.dot(desc_embeddings, desc_embeddings.T)
    similar_desc_pairs = np.argwhere(desc_similarities > 0.8)  # High text similarity threshold
    
    # Create negative pairs based on similar descriptions
    for i, j in similar_desc_pairs:
        if i < j:  # Avoid duplicates
            text1, text2 = all_descriptions[i], all_descriptions[j]
            concept1, concept2 = desc_to_concept[text1], desc_to_concept[text2]
            
            if concept1 != concept2:
                desc_similar_negatives.append(InputExample(texts=[text1, text2], label=0.0))
    
    # Create negative pairs based on concept name similarity
    for (concept1, concept2), sim in concept_similarities.items():
        if concept1 != concept2:
            texts1 = []
            texts2 = []
            # Collect texts for both concepts across all domains
            for domain in main_dict:
                if concept1 in main_dict[domain]:
                    texts1.extend(main_dict[domain][concept1])
                if concept2 in main_dict[domain]:
                    texts2.extend(main_dict[domain][concept2])
            
            # Create pairs between similar concepts
            for text1 in texts1[:2]:  # Limit number of pairs per concept pair
                for text2 in texts2[:2]:
                    concept_similar_negatives.append(InputExample(texts=[text1, text2], label=0.0))
    
    # Balance the dataset
    total_positives = len(positive_pairs)
    total_negatives = total_positives  # Keep total negatives equal to positives
    
    # Aim for roughly equal split between the two types of negatives
    target_per_type = total_negatives // 2
    
    # Sample negative pairs
    concept_similar_negatives = random.sample(
        concept_similar_negatives, 
        min(target_per_type, len(concept_similar_negatives))
    )
    
    desc_similar_negatives = random.sample(
        desc_similar_negatives,
        min(total_negatives - len(concept_similar_negatives), len(desc_similar_negatives))
    )
    
    # Combine all pairs
    all_pairs = positive_pairs + concept_similar_negatives + desc_similar_negatives
    random.shuffle(all_pairs)
    
    # Log statistics
    logger.info(f"Created {len(positive_pairs)} positive pairs")
    logger.info(f"Created {len(concept_similar_negatives)} concept-similar negative pairs")
    logger.info(f"Created {len(desc_similar_negatives)} description-similar negative pairs")
    logger.info(f"Total pairs: {len(all_pairs)}")
    
    return all_pairs











def create_pairs(self, df: pd.DataFrame) -> List[InputExample]:
    """Create training pairs for contrastive learning with enhanced negative pair strategy."""
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
    same_domain_negatives = []
    other_domain_negatives = []
    
    # Create positive pairs (same concept)
    for domain in main_dict:
        for concept in main_dict[domain]:
            texts = main_dict[domain][concept]
            for text1, text2 in combinations(texts, 2):
                positive_pairs.append(InputExample(texts=[text1, text2], label=1.0))

    # Calculate concept name similarities using sentence transformer
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    all_concepts = list(set(concept for domain in main_dict for concept in main_dict[domain]))
    concept_embeddings = model.encode(all_concepts)
    concept_embeddings = concept_embeddings / np.linalg.norm(concept_embeddings, axis=1)[:, np.newaxis]
    
    concept_similarities = {}
    for i, j in combinations(range(len(all_concepts)), 2):
        sim = float(np.dot(concept_embeddings[i], concept_embeddings[j]))
        concept_similarities[(all_concepts[i], all_concepts[j])] = sim
        concept_similarities[(all_concepts[j], all_concepts[i])] = sim
    
    # Create negative pairs with weights based on concept similarity
    all_texts = [(text, domain, concept) 
                for domain in main_dict 
                for concept in main_dict[domain] 
                for text in main_dict[domain][concept]]
    
    for (text1, domain1, concept1), (text2, domain2, concept2) in combinations(all_texts, 2):
        if concept1 != concept2:  # Different concepts
            similarity = concept_similarities.get((concept1, concept2), 0.0)
            weight = 1.0 + similarity  # Higher weight for similar concepts
            
            if domain1 == domain2:  # Same domain
                same_domain_negatives.append(
                    InputExample(texts=[text1, text2], label=0.0))
            else:  # Different domain
                other_domain_negatives.append(
                    InputExample(texts=[text1, text2], label=0.0))

    # Balance negative pairs
    total_positives = len(positive_pairs)
    num_same_domain = int(total_positives * 0.7)  # 70% same domain negatives
    num_other_domain = int(total_positives * 0.3)  # 30% other domain negatives
    
    # Ensure we don't sample more than available
    num_same_domain = min(num_same_domain, len(same_domain_negatives))
    num_other_domain = min(num_other_domain, len(other_domain_negatives))
    
    # Sample negative pairs
    sampled_same_domain = random.sample(same_domain_negatives, num_same_domain)
    sampled_other_domain = random.sample(other_domain_negatives, num_other_domain)
    
    # Combine all pairs
    all_pairs = positive_pairs + sampled_same_domain + sampled_other_domain
    random.shuffle(all_pairs)
    
    logger.info(f"Created {len(positive_pairs)} positive pairs")
    logger.info(f"Created {num_same_domain} same-domain negative pairs")
    logger.info(f"Created {num_other_domain} other-domain negative pairs")
    logger.info(f"Total pairs: {len(all_pairs)}")
    
    return all_pairs








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
