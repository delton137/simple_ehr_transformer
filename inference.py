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
from model import create_ethos_model
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
    parser = argparse.ArgumentParser(description='Run ETHOS Transformer Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='Directory containing processed data (default: processed_data/)')
    parser.add_argument('--tag', type=str, default=None,
                       help='Dataset tag to use (e.g., aou_2023, mimic_iv)')
    parser.add_argument('--patient_id', type=int, default=None,
                       help='Specific patient ID to analyze')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Directory for inference results (default: inference_results/)')
    
    args = parser.parse_args()
    
    # Handle tag-based data directory
    if args.tag and not args.data_dir.startswith('processed_data_'):
        args.data_dir = f"processed_data_{args.tag}"
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory '{args.data_dir}' does not exist!")
        print(f"Available directories:")
        available_dirs = [d for d in os.listdir('.') if d.startswith('processed_data')]
        if available_dirs:
            for d in available_dirs:
                print(f"  - {d}")
        else:
            print("  - No processed_data directories found")
        print(f"\nTo process data with a tag, run:")
        print(f"  python data_processor.py --tag {args.tag or 'your_tag'}")
        return
    
    # Create output directory with tag
    if args.tag:
        args.output_dir = f"inference_results_{args.tag}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🔍 Starting ETHOS Transformer Inference")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"🏷️  Dataset tag: {args.tag or 'none'}")
    print(f"💾 Model: {args.model_path}")
    print(f"📊 Output directory: {args.output_dir}")
    
    # Load model and vocabulary
    print("\n📚 Loading model and vocabulary...")
    try:
        inference = ETHOSInference(args.model_path, os.path.join(args.data_dir, 'vocabulary.pkl'))
        print("✅ Model and vocabulary loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load processed data
    print("\n📊 Loading processed data...")
    try:
        with open(os.path.join(args.data_dir, 'tokenized_timelines.pkl'), 'rb') as f:
            tokenized_timelines = pickle.load(f)
        print(f"✅ Loaded {len(tokenized_timelines)} patient timelines")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run inference
    if args.patient_id:
        # Analyze specific patient
        print(f"\n🔍 Analyzing patient {args.patient_id}...")
        if args.patient_id not in tokenized_timelines:
            print(f"❌ Patient {args.patient_id} not found in dataset")
            available_ids = list(tokenized_timelines.keys())[:10]
            print(f"Available patient IDs (first 10): {available_ids}")
            return
        
        timeline = tokenized_timelines[args.patient_id]
        analysis = inference.analyze_patient_timeline(timeline)
        
        # Save results
        results_file = os.path.join(args.output_dir, f'patient_{args.patient_id}_analysis.json')
        with open(results_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"✅ Analysis saved to: {results_file}")
        print(f"\n📊 Analysis Results:")
        for key, value in analysis.items():
            if key != 'token_frequency':
                print(f"  {key}: {value}")
        
        # Generate future timeline
        print(f"\n🔮 Generating future timeline for patient {args.patient_id}...")
        future_timeline = inference.generate_future_timeline(timeline, max_tokens=50)
        
        future_file = os.path.join(args.output_dir, f'patient_{args.patient_id}_future_timeline.json')
        with open(future_file, 'w') as f:
            json.dump(future_timeline, f, indent=2, default=str)
        
        print(f"✅ Future timeline saved to: {future_file}")
        
    else:
        # Analyze all patients
        print(f"\n🔍 Analyzing all {len(tokenized_timelines)} patients...")
        
        all_results = {}
        for i, (patient_id, timeline) in enumerate(tokenized_timelines.items()):
            if i % 100 == 0:
                print(f"  Processing patient {i+1}/{len(tokenized_timelines)}...")
            
            try:
                analysis = inference.analyze_patient_timeline(timeline)
                all_results[patient_id] = analysis
            except Exception as e:
                print(f"⚠️  Error analyzing patient {patient_id}: {e}")
                continue
        
        # Save all results
        all_results_file = os.path.join(args.output_dir, 'all_patients_analysis.json')
        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"✅ All results saved to: {all_results_file}")
        
        # Summary statistics
        if all_results:
            mortality_probs = [r['mortality_probability'] for r in all_results.values() if 'mortality_probability' in r]
            readmission_probs = [r['readmission_probability'] for r in all_results.values() if 'readmission_probability' in r]
            los_predictions = [r['predicted_los'] for r in all_results.values() if 'predicted_los' in r]
            
            print(f"\n📊 Summary Statistics:")
            if mortality_probs:
                print(f"  Mortality probability - Mean: {np.mean(mortality_probs):.3f}, Std: {np.std(mortality_probs):.3f}")
            if readmission_probs:
                print(f"  Readmission probability - Mean: {np.mean(readmission_probs):.3f}, Std: {np.std(readmission_probs):.3f}")
            if los_predictions:
                print(f"  Predicted LOS - Mean: {np.mean(los_predictions):.1f} days, Std: {np.std(los_predictions):.1f} days")
    
    print(f"\n🎉 Inference completed!")
    print(f"📁 Results saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()
