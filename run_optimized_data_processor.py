#!/usr/bin/env python3
"""
Example script to run the optimized data_processor_random_forest.py with performance monitoring.
"""

import subprocess
import time
import psutil
import os
import sys

def monitor_performance():
    """Monitor CPU and memory usage during execution."""
    process = psutil.Process()
    cpu_percent = process.cpu_percent()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    return cpu_percent, memory_mb

def run_optimized_processor(data_dir, target_concept_id, n_processes=None, chunk_size=100):
    """Run the optimized data processor with performance monitoring."""
    
    # Build command
    cmd = [
        sys.executable, 
        'data_processor_random_forest.py',
        '--data_dir', data_dir,
        '--target_concept_id', str(target_concept_id),
        '--sparse',  # Use sparse matrices to save memory
        '--chunk_size', str(chunk_size)
    ]
    
    if n_processes:
        cmd.extend(['--n_processes', str(n_processes)])
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Available CPUs: {psutil.cpu_count()}")
    print(f"Available RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print("-" * 60)
    
    start_time = time.time()
    start_cpu, start_memory = monitor_performance()
    
    try:
        # Run the process
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        end_time = time.time()
        end_cpu, end_memory = monitor_performance()
        
        print("=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        print(f"Peak memory usage: {end_memory:.1f} MB")
        print(f"Average CPU usage: {(start_cpu + end_cpu) / 2:.1f}%")
        print("=" * 60)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_optimized_data_processor.py <data_dir> <target_concept_id> [n_processes] [chunk_size]")
        print("Example: python run_optimized_data_processor.py processed_data_aou_2021_2022/train 12345 8 200")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    target_concept_id = int(sys.argv[2])
    n_processes = int(sys.argv[3]) if len(sys.argv) > 3 else None
    chunk_size = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    
    success = run_optimized_processor(data_dir, target_concept_id, n_processes, chunk_size)
    sys.exit(0 if success else 1)
