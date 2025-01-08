class ImprovedSampleGenerator:
    """Enhanced sample generator with better parallelization"""
    
    def generate_batch(self,
                      domain: str,
                      concept: str,
                      context_samples: List[Dict],
                      definition: str = "") -> List[Dict]:
        """Generate a single batch of samples (non-async version)"""
        prompt = self.generate_prompt(domain, concept, context_samples, self.batch_size, definition)
        
        try:
            self.rate_limiter.wait()
            response = chinou_response(prompt)  # Your LLM call
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
        """Generate samples with parallel processing"""
        
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
                        # Cancel remaining futures
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
                
                # Clean up futures
                futures = [f for f in futures if not f.done()]
        
        # Final progress log
        stats.log_progress(domain, concept)
        return synthetic_samples[:num_needed]
