import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import debug
from src.runner.runner import Runner

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = argparse.ArgumentParser(description="Run jailbreak testing on the FigStep dataset.")
    parser.add_argument("--config_dir", type=str, default="../../configs/", help="Path to the configuration directory.")
    parser.add_argument("--config_name", type=str, default="llava_PT_training.yaml", help="Name of the configuration file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    if args.debug:
        debug()

    # Initialize the Runner and start training
    runner = Runner(args.config_dir, args.config_name)
    runner.train()

if __name__ == "__main__":
    main()