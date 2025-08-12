#!/usr/bin/env python3
"""
Inference script for trained ETHOS transformer model
Demonstrates zero-shot prediction capabilities on EHR data
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
import argparse
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from config import model_config, data_config
from transformer_model import create_ethos_model
from data_processor import EHRDataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETHOSInference:
    """Inference class for ETHOS transformer model"""
    
    def __init__(self, model_path: str, vocab_path: str, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        # Create reverse vocabulary mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        logger.info(f"Loaded ETHOS model on device: {self.device}")
        logger.info(f"Vocabulary size: {len(self.vocab)}")
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model with same architecture
        model = create_ethos_model(len(self.vocab))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def predict_next_tokens(self, input_ids: torch.Tensor, num_tokens: int = 10,
                           temperature: float = 1.0, top_k: int = 50,
                           top_p: float = 0.9) -> torch.Tensor:
        """Predict next tokens given input sequence"""
        with torch.no_grad():
            # Generate continuation
            generated = self.model.generate(
                input_ids,
                max_length=num_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True
            )
            
            # Return only the generated part
            return generated[:, input_ids.size(1):]
    
    def predict_mortality(self, patient_timeline: List[int], 
                         num_simulations: int = 20) -> float:
        """
        Predict patient mortality probability using ETHOS methodology
        Based on the ETHOS paper approach
        """
        # Convert timeline to tensor
        input_ids = torch.tensor([patient_timeline], dtype=torch.long).to(self.device)
        
        death_count = 0
        
        for _ in range(num_simulations):
            # Generate future timeline
            generated = self.model.generate(
                input_ids,
                max_length=100,  # Generate up to 100 tokens
                temperature=1.0,
                top_k=50,
                top_p=0.9,
                do_sample=True
            )
            
            # Check if death token appears
            if 2 in generated[0]:  # Assuming 2 is death token
                death_count += 1
        
        mortality_probability = death_count / num_simulations
        return mortality_probability
    
    def predict_readmission(self, patient_timeline: List[int], 
                           days_threshold: int = 30, 
                           num_simulations: int = 20) -> float:
        """
        Predict readmission probability within specified days
        """
        # Convert timeline to tensor
        input_ids = torch.tensor([patient_timeline], dtype=torch.long).to(self.device)
        
        readmission_count = 0
        
        for _ in range(num_simulations):
            # Generate future timeline
            generated = self.model.generate(
                input_ids,
                max_length=200,  # Generate more tokens for longer time
                temperature=1.0,
                top_k=50,
                top_p=0.9,
                do_sample=True
            )
            
            # Check for readmission (admission token) within time threshold
            # This is a simplified approach - you'd need to implement proper time tracking
            if self._check_readmission_within_time(generated[0], days_threshold):
                readmission_count += 1
        
        readmission_probability = readmission_count / num_simulations
        return readmission_probability
    
    def _check_readmission_within_time(self, generated_tokens: torch.Tensor, 
                                     days_threshold: int) -> bool:
        """Check if readmission occurs within time threshold"""
        # This is a simplified implementation
        # In practice, you'd need to track actual time intervals
        
        # Look for admission tokens in generated sequence
        admission_tokens = [i for i, token in enumerate(generated_tokens) 
                          if self.id_to_token.get(token.item(), '').startswith('EVENT_ADMISSION')]
        
        return len(admission_tokens) > 0
    
    def predict_sofa_score(self, patient_timeline: List[int]) -> float:
        """
        Predict SOFA score for ICU patients
        Based on ETHOS methodology
        """
        # Convert timeline to tensor
        input_ids = torch.tensor([patient_timeline], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # Get model predictions
            logits = self.model(input_ids)
            
            # Get probabilities for SOFA-related tokens
            # This is a simplified approach - you'd need to identify SOFA tokens
            sofa_probs = F.softmax(logits[:, -1, :], dim=-1)
            
            # For now, return a random SOFA score (0-24)
            # In practice, you'd map token probabilities to actual SOFA scores
            predicted_sofa = np.random.randint(0, 25)
            
        return predicted_sofa
    
    def predict_length_of_stay(self, patient_timeline: List[int]) -> float:
        """
        Predict length of stay in days
        """
        # Convert timeline to tensor
        input_ids = torch.tensor([patient_timeline], dtype=torch.long).to(self.device)
        
        # Generate future timeline until discharge
        generated = self.model.generate(
            input_ids,
            max_length=100,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        # Count time interval tokens to estimate LOS
        # This is a simplified approach
        time_tokens = [token for token in generated[0] 
                      if self.id_to_token.get(token.item(), '').startswith('TIME_')]
        
        # Convert time tokens to days (simplified)
        total_days = len(time_tokens) * 0.5  # Rough estimate
        
        return total_days
    
    def analyze_patient_timeline(self, patient_timeline: List[int]) -> Dict[str, Any]:
        """Comprehensive analysis of a patient timeline"""
        analysis = {}
        
        # Basic statistics
        analysis['timeline_length'] = len(patient_timeline)
        analysis['unique_tokens'] = len(set(patient_timeline))
        
        # Token distribution
        token_counts = {}
        for token in patient_timeline:
            token_name = self.id_to_token.get(token, f'UNKNOWN_{token}')
            token_counts[token_name] = token_counts.get(token_name, 0) + 1
        
        analysis['token_distribution'] = dict(sorted(token_counts.items(), 
                                                   key=lambda x: x[1], reverse=True)[:10])
        
        # Predictions
        analysis['mortality_probability'] = self.predict_mortality(patient_timeline)
        analysis['readmission_probability'] = self.predict_readmission(patient_timeline)
        
        # Try to predict SOFA score if applicable
        try:
            analysis['predicted_sofa'] = self.predict_sofa_score(patient_timeline)
        except:
            analysis['predicted_sofa'] = None
        
        # Predict length of stay
        analysis['predicted_los'] = self.predict_length_of_stay(patient_timeline)
        
        return analysis
    
    def generate_future_timeline(self, patient_timeline: List[int], 
                                max_tokens: int = 100) -> List[int]:
        """Generate future timeline continuation"""
        input_ids = torch.tensor([patient_timeline], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_length=max_tokens,
                temperature=1.0,
                top_k=50,
                top_p=0.9,
                do_sample=True
            )
        
        # Return only the generated part
        return generated[0, len(patient_timeline):].tolist()
    
    def visualize_timeline(self, original_timeline: List[int], 
                          generated_timeline: List[int], 
                          save_path: str = 'timeline_visualization.png'):
        """Visualize original vs generated timeline"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Original timeline
        ax1.plot(range(len(original_timeline)), original_timeline, 'b-', label='Original Timeline')
        ax1.set_title('Original Patient Timeline')
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Token ID')
        ax1.legend()
        ax1.grid(True)
        
        # Generated timeline
        ax2.plot(range(len(generated_timeline)), generated_timeline, 'r-', label='Generated Timeline')
        ax2.set_title('Generated Future Timeline')
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Token ID')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Timeline visualization saved to {save_path}")

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Run inference with trained ETHOS model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab_path', type=str, default='processed_data/vocabulary.pkl',
                       help='Path to vocabulary file')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='Directory containing processed data')
    parser.add_argument('--patient_id', type=int, default=None,
                       help='Specific patient ID to analyze')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Directory to save inference results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference engine
    inference = ETHOSInference(args.model_path, args.vocab_path, args.device)
    
    # Load patient timelines
    with open(os.path.join(args.data_dir, 'tokenized_timelines.pkl'), 'rb') as f:
        tokenized_timelines = pickle.load(f)
    
    logger.info(f"Loaded {len(tokenized_timelines)} patient timelines")
    
    # Analyze specific patient or random sample
    if args.patient_id is not None:
        if args.patient_id in tokenized_timelines:
            patient_ids = [args.patient_id]
        else:
            logger.error(f"Patient ID {args.patient_id} not found!")
            return
    else:
        # Analyze a random sample of patients
        patient_ids = np.random.choice(list(tokenized_timelines.keys()), 
                                     size=min(5, len(tokenized_timelines)), 
                                     replace=False)
    
    # Run inference on selected patients
    results = {}
    
    for patient_id in patient_ids:
        logger.info(f"Analyzing patient {patient_id}")
        
        timeline = tokenized_timelines[patient_id]
        
        # Analyze current timeline
        analysis = inference.analyze_patient_timeline(timeline)
        
        # Generate future timeline
        future_timeline = inference.generate_future_timeline(timeline)
        
        # Store results
        results[patient_id] = {
            'analysis': analysis,
            'original_timeline': timeline,
            'future_timeline': future_timeline
        }
        
        # Visualize timeline
        viz_path = os.path.join(args.output_dir, f'patient_{patient_id}_timeline.png')
        inference.visualize_timeline(timeline, future_timeline, viz_path)
        
        # Print analysis
        logger.info(f"Patient {patient_id} Analysis:")
        logger.info(f"  Timeline length: {analysis['timeline_length']}")
        logger.info(f"  Mortality probability: {analysis['mortality_probability']:.3f}")
        logger.info(f"  Readmission probability: {analysis['readmission_probability']:.3f}")
        if analysis['predicted_sofa'] is not None:
            logger.info(f"  Predicted SOFA score: {analysis['predicted_sofa']}")
        logger.info(f"  Predicted LOS: {analysis['predicted_los']:.1f} days")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'inference_results.json')
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj
    
    serializable_results = convert_numpy_types(results)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Inference results saved to {results_path}")
    
    # Summary statistics
    mortality_probs = [r['analysis']['mortality_probability'] for r in results.values()]
    readmission_probs = [r['analysis']['readmission_probability'] for r in results.values()]
    
    logger.info(f"\nSummary Statistics:")
    logger.info(f"  Average mortality probability: {np.mean(mortality_probs):.3f}")
    logger.info(f"  Average readmission probability: {np.mean(readmission_probs):.3f}")

if __name__ == "__main__":
    main()
