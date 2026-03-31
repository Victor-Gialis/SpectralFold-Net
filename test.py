#!/usr/bin/env python3
"""
SAP Backbone Evaluation Suite

Comprehensive evaluation of Self-Attention Pretraining (SAP) backbone across
different data regimes and configurations.

Performs:
  1. Pretraining: SAP models with varying downsampling factors
  2. Downstream evaluation: Classification tasks with different data ratios,
     finetuning options, and multiple random seeds
"""

import os
import sys
import argparse
import logging
from itertools import product
from datetime import datetime
from pathlib import Path

# Set PYTHONPATH to include current directory
pythonpath = os.getenv("PYTHONPATH", "")
if pythonpath:
    pythonpath = pythonpath + ":" + os.path.abspath(".")
else:
    pythonpath = os.path.abspath(".")
os.environ["PYTHONPATH"] = pythonpath

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"sap_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SAP backbone evaluation experiments"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--pretrain_dataset",
        default="CWRU",
        choices=["CWRU", "LASPI"],
        help="Dataset for pretraining"
    )
    parser.add_argument(
        "--downstream_dataset",
        default="CWRU",
        choices=["CWRU", "LASPI"],
        help="Dataset for downstream evaluation"
    )
    
    # Dataloader configuration
    parser.add_argument("--window_size", type=int, default=2048)
    parser.add_argument("--window_stride", type=int, default=512)
    
    # Experiment ranges
    parser.add_argument(
        "--downsampling_factors",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="Downsampling factors to evaluate"
    )
    parser.add_argument(
        "--data_ratios",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.1, 0.2],
        help="Data ratios for downstream evaluation"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds for reproducibility"
    )
    
    # Training configuration
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--pretrain_batch_size", type=int, default=256)
    parser.add_argument("--downstream_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.0003695)
    parser.add_argument("--weight_decay", type=float, default=1.1133e-5)
    
    # Model configuration
    parser.add_argument(
        "--head_type",
        default="linear",
        choices=["linear", "nonlinear"],
        help="Type of classification head"
    )
    parser.add_argument(
        "--split_type",
        default="speed_load_stratified",
        choices=["independent", "speed_stratified", "speed_load_stratified", "sample_stratified"],
        help="Data split strategy"
    )
    
    # Finetuning options
    parser.add_argument(
        "--finetune_both",
        action="store_true",
        default=False,
        help="Run both with and without finetuning"
    )
    
    return parser.parse_args()


def get_downstream_epochs(data_ratio):
    """Determine number of epochs based on data ratio."""
    if data_ratio <= 0.01:
        return 100
    elif data_ratio <= 0.05:
        return 50
    else:
        return 30


def run_command(cmd, description):
    """Execute command and log results."""
    logger.info(f"\n{'='*80}")
    logger.info(f"📋 {description}")
    logger.info(f"{'='*80}")
    logger.info(f"Command: {cmd}")
    
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        logger.error(f"❌ Command failed with exit code {exit_code}")
        return False
    else:
        logger.info(f"✅ Command completed successfully")
        return True


def main():
    """Run the complete evaluation suite."""
    args = parse_arguments()
    
    logger.info("="*80)
    logger.info("🚀 Starting SAP Backbone Evaluation Suite")
    logger.info("="*80)
    logger.info(f"Pretrain dataset: {args.pretrain_dataset}")
    logger.info(f"Downstream dataset: {args.downstream_dataset}")
    logger.info(f"Window size: {args.window_size}")
    logger.info(f"Downsampling factors: {args.downsampling_factors}")
    logger.info(f"Data ratios: {args.data_ratios}")
    logger.info(f"Seeds: {args.seeds}")
    
    total_experiments = 0
    successful_experiments = 0
    failed_experiments = 0
    
    # Determine finetuning options
    finetune_options = [True, False] if args.finetune_both else [False]
    
    for downsampling_factor in args.downsampling_factors:
        # =========================
        # Pretraining experiments
        # =========================
        pretrain_cmd = (
            f"python experiments/pretrain_sap.py "
            f"--pretrain_dataset {args.pretrain_dataset} "
            f"--batch_size {args.pretrain_batch_size} "
            f"--window_size {args.window_size} "
            f"--window_stride {args.window_stride} "
            f"--downsampling_factor {downsampling_factor} "
            f"--epochs {args.pretrain_epochs}"
        )
        
        description = f"SAP Pretraining (downsampling={downsampling_factor})"
        success = run_command(pretrain_cmd, description)
        total_experiments += 1
        
        if not success:
            logger.warning(f"⚠️ Skipping downstream experiments for downsampling_factor={downsampling_factor}")
            failed_experiments += 1
            continue
        
        successful_experiments += 1
        
        # =========================
        # Downstream experiments
        # =========================
        downstream_count = 0
        downstream_success = 0
        
        for data_ratio, finetune_option, seed in product(
            args.data_ratios, 
            finetune_options, 
            args.seeds
        ):
            downstream_epochs = get_downstream_epochs(data_ratio)
            
            downstream_cmd = (
                f"python experiments/downstream_sap.py "
                f"--pretrain_dataset {args.pretrain_dataset} "
                f"--downstream_dataset {args.downstream_dataset} "
                f"--batch_size {args.downstream_batch_size} "
                f"--window_size {args.window_size} "
                f"--window_stride {args.window_stride} "
                f"--data_ratio {data_ratio} "
                f"--split_type {args.split_type} "
                f"--learning_rate {args.learning_rate} "
                f"--weight_decay {args.weight_decay} "
                f"--epochs {downstream_epochs} "
                f"--head_type {args.head_type} "
                f"--task classification "
                f"--seed {seed}"
            )
            
            if finetune_option:
                downstream_cmd += " --finetune"
            
            downstream_count += 1
            total_experiments += 1
            
            finetune_str = "with finetune" if finetune_option else "frozen"
            description = (
                f"Downstream evaluation (downsampling={downsampling_factor}, "
                f"data_ratio={data_ratio}, seed={seed}, {finetune_str})"
            )
            
            success = run_command(downstream_cmd, description)
            if success:
                successful_experiments += 1
                downstream_success += 1
            else:
                failed_experiments += 1
        
        logger.info(f"✅ Downstream results: {downstream_success}/{downstream_count} successful")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("📊 EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Successful: {successful_experiments} ✅")
    logger.info(f"Failed: {failed_experiments} ❌")
    logger.info(f"Success rate: {100*successful_experiments/total_experiments:.1f}%")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    return 0 if failed_experiments == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

