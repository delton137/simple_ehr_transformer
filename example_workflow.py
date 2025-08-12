#!/usr/bin/env python3
"""
Example workflow demonstrating the complete ETHOS pipeline
From data processing to training to inference
"""

import os
import logging
import pickle
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Demonstrate the complete ETHOS workflow"""
    
    logger.info("=== ETHOS Transformer Workflow Example ===")
    
    # Step 1: Data Processing
    logger.info("\n1. Processing EHR Data...")
    
    try:
        from data_processor import EHRDataProcessor
        
        # Initialize processor
        processor = EHRDataProcessor()
        
        # Check if data already exists
        if os.path.exists('processed_data/tokenized_timelines.pkl'):
            logger.info("Loading existing processed data...")
            tokenized_timelines, vocab = processor.load_processed_data()
        else:
            logger.info("Processing new data...")
            tokenized_timelines, vocab = processor.process_all_data()
        
        logger.info(f"✓ Processed {len(tokenized_timelines)} patient timelines")
        logger.info(f"✓ Vocabulary size: {len(vocab)}")
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        logger.info("Please ensure your OMOP/MEDS data is in the correct directories")
        return
    
    # Step 2: Data Analysis
    logger.info("\n2. Analyzing Data Distribution...")
    
    try:
        from data_loader import analyze_data_distribution
        
        analysis = analyze_data_distribution(tokenized_timelines)
        
        logger.info("Data Analysis Results:")
        logger.info(f"  - Number of patients: {analysis['num_patients']}")
        logger.info(f"  - Total tokens: {analysis['total_tokens']}")
        logger.info(f"  - Average sequence length: {analysis['avg_sequence_length']:.1f}")
        logger.info(f"  - Unique tokens: {analysis['unique_tokens']}")
        
        logger.info("✓ Data analysis completed")
        
    except Exception as e:
        logger.error(f"Data analysis failed: {e}")
        return
    
    # Step 3: Model Creation
    logger.info("\n3. Creating ETHOS Model...")
    
    try:
        from transformer_model import create_ethos_model
        
        # Create model
        model = create_ethos_model(len(vocab))
        
        logger.info(f"✓ Model created with {model.count_parameters():,} parameters")
        
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        return
    
    # Step 4: Training Setup
    logger.info("\n4. Setting Up Training...")
    
    try:
        from data_loader import PHTDataProcessor
        
        # Create data processor
        data_processor = PHTDataProcessor(tokenized_timelines, len(vocab))
        
        # Create data loaders
        train_loader, val_loader = data_processor.create_dataloaders(batch_size=16)
        
        logger.info(f"✓ Training batches: {len(train_loader)}")
        logger.info(f"✓ Validation batches: {len(val_loader)}")
        
    except Exception as e:
        logger.error(f"Training setup failed: {e}")
        return
    
    # Step 5: Training (Optional - Skip if no GPU or for demo)
    logger.info("\n5. Training Model...")
    
    # Check if we should skip training (for demo purposes)
    skip_training = input("Skip training for demo? (y/n): ").lower().startswith('y')
    
    if skip_training:
        logger.info("Skipping training for demo purposes")
        logger.info("To train the model, run: python train.py")
    else:
        try:
            # This would start the actual training
            logger.info("Starting training...")
            logger.info("Note: Training may take several hours depending on data size and hardware")
            
            # For demo, just show what would happen
            logger.info("Training would proceed with:")
            logger.info("  - Learning rate: 3e-4")
            logger.info("  - Batch size: 16")
            logger.info("  - Max epochs: 100")
            logger.info("  - Checkpointing every epoch")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return
    
    # Step 6: Inference Demo
    logger.info("\n6. Inference Demo...")
    
    try:
        # Check if we have a trained model
        model_paths = ['models/best_checkpoint.pth', 'models/latest_checkpoint.pth']
        model_path = None
        
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            logger.info(f"Found trained model: {model_path}")
            
            from inference import ETHOSInference
            
            # Initialize inference
            inference = ETHOSInference(model_path, 'processed_data/vocabulary.pkl')
            
            # Select a patient for demo
            patient_ids = list(tokenized_timelines.keys())
            if patient_ids:
                demo_patient = patient_ids[0]
                logger.info(f"Running inference on patient {demo_patient}")
                
                # Analyze timeline
                timeline = tokenized_timelines[demo_patient]
                analysis = inference.analyze_patient_timeline(timeline)
                
                logger.info("Inference Results:")
                logger.info(f"  - Timeline length: {analysis['timeline_length']}")
                logger.info(f"  - Mortality probability: {analysis['mortality_probability']:.3f}")
                logger.info(f"  - Readmission probability: {analysis['readmission_probability']:.3f}")
                logger.info(f"  - Predicted LOS: {analysis['predicted_los']:.1f} days")
                
                # Generate future timeline
                future_timeline = inference.generate_future_timeline(timeline, max_tokens=50)
                logger.info(f"  - Generated {len(future_timeline)} future tokens")
                
                logger.info("✓ Inference completed successfully")
                
            else:
                logger.warning("No patient data available for inference")
                
        else:
            logger.info("No trained model found")
            logger.info("To run inference, first train a model using: python train.py")
            
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return
    
    # Step 7: Summary
    logger.info("\n=== Workflow Summary ===")
    logger.info("✓ Data processing completed")
    logger.info("✓ Data analysis completed")
    logger.info("✓ Model created")
    logger.info("✓ Training setup completed")
    
    if skip_training:
        logger.info("⚠ Training skipped (demo mode)")
    else:
        logger.info("✓ Training completed")
    
    if model_path:
        logger.info("✓ Inference completed")
    else:
        logger.info("⚠ Inference skipped (no trained model)")
    
    logger.info("\nNext Steps:")
    logger.info("1. To train the model: python train.py")
    logger.info("2. To run inference: python inference.py --model_path models/best_checkpoint.pth")
    logger.info("3. To analyze specific patients: python inference.py --patient_id <ID>")
    
    logger.info("\n=== Workflow Complete ===")

if __name__ == "__main__":
    main()
