#!/usr/bin/env python3
"""
Script to convert FSDP checkpoint back to a complete HuggingFace model format.

This script uses the VERL model merger to convert distributed FSDP checkpoints
into a standard HuggingFace format that can be loaded and used for inference.

Usage:
    python convert_checkpoint.py --checkpoint_path /path/to/checkpoint --output_dir /path/to/output [--base_model_path /path/to/base/model]

Example:
    python convert_checkpoint.py \
        --checkpoint_path "/home/wangyu/ckpt/mem-alpha-grpo-memory-agent-qwen3-4b-no-compression-no-thinking-8node/global_step_40" \
        --output_dir "/home/wangyu/converted_models/mem-alpha-grpo-qwen3-4b-step40-8node" \
        --base_model_path "Qwen/Qwen3-4B"
"""

import argparse
import os
import sys
from pathlib import Path

# Add the current working directory to Python path to import verl modules
sys.path.append(os.getcwd())

try:
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger
    from verl.model_merger.base_model_merger import ModelMergerConfig
except ImportError as e:
    print(f"Error importing VERL modules: {e}")
    print("Make sure you're running this script from the Mem-alpha directory")
    sys.exit(1)


def validate_checkpoint_path(checkpoint_path):
    """Validate that the checkpoint path contains the expected FSDP structure."""
    actor_path = Path(checkpoint_path) / "actor"
    if not actor_path.exists():
        raise ValueError(f"Actor directory not found in checkpoint: {actor_path}")
    
    # Check for FSDP model files
    model_files = list(actor_path.glob("model_world_size_*_rank_*.pt"))
    if not model_files:
        raise ValueError(f"No FSDP model files found in: {actor_path}")
    
    # Check for FSDP config
    fsdp_config = actor_path / "fsdp_config.json"
    if not fsdp_config.exists():
        print(f"Warning: FSDP config not found at {fsdp_config}")
    
    # Check for HuggingFace config directory
    hf_dir = actor_path / "huggingface"
    if not hf_dir.exists():
        raise ValueError(f"HuggingFace config directory not found: {hf_dir}")
    
    config_file = hf_dir / "config.json"
    if not config_file.exists():
        raise ValueError(f"HuggingFace config.json not found: {config_file}")
    
    print(f"✓ Validated checkpoint structure at: {checkpoint_path}")
    print(f"✓ Found {len(model_files)} FSDP model shards")
    return str(actor_path), str(hf_dir)


def create_model_merger_config(actor_path, hf_config_path, output_dir, base_model_path=None):
    """Create configuration for the model merger."""
    
    # If base_model_path is not provided, use the HuggingFace config path from checkpoint
    if base_model_path is None:
        base_model_path = hf_config_path
        print(f"Using HuggingFace config from checkpoint: {base_model_path}")
    else:
        print(f"Using provided base model path: {base_model_path}")
    
    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=actor_path,
        target_dir=output_dir,
        hf_model_config_path=base_model_path,
        tie_word_embedding=False,  # Qwen models typically don't tie embeddings
        is_value_model=False,      # This is not a value model
        use_cpu_initialization=True  # Use CPU to handle large models
    )
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to HuggingFace model format")
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True,
        help="Path to the checkpoint directory (e.g., /path/to/global_step_40)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save the converted HuggingFace model"
    )
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        default=None,
        help="Path to the original base model for config (optional, defaults to checkpoint's huggingface dir)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.checkpoint_path).exists():
        print(f"Error: Checkpoint path does not exist: {args.checkpoint_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Validate checkpoint structure and get paths
        actor_path, hf_config_path = validate_checkpoint_path(args.checkpoint_path)
        
        # Create merger configuration
        config = create_model_merger_config(
            actor_path=actor_path,
            hf_config_path=hf_config_path,
            output_dir=args.output_dir,
            base_model_path=args.base_model_path
        )
        
        print(f"\nStarting model conversion...")
        print(f"Source: {actor_path}")
        print(f"Target: {args.output_dir}")
        print(f"Config: {config.hf_model_config_path}")
        
        # Initialize and run the merger
        merger = FSDPModelMerger(config)
        merger.merge_and_save()
        
        print(f"\n✓ Model conversion completed successfully!")
        print(f"✓ Converted model saved to: {args.output_dir}")
        print(f"\nYou can now load the model using:")
        print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"model = AutoModelForCausalLM.from_pretrained('{args.output_dir}')")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()