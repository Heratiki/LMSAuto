# LMSAuto Command Line Interface
# Entry point for the application

import argparse
import logging
import sys

from lmsauto.shared.context import SharedContext
from lmsauto.profiler import SystemProfiler
from lmsauto.scanner import ModelScanner, LMStudioScanner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Update root logger level
    logging.getLogger().setLevel(numeric_level)
    logger.setLevel(numeric_level)

def main():
    """Main function for the LMSAuto CLI."""
    parser = argparse.ArgumentParser(
        description="LMSAuto - LM Studio Autonomous Model Settings Optimizer"
    )
    
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan for models without generating profiles"
    )
    
    parser.add_argument(
        "--profile-system",
        action="store_true",
        help="Profile system hardware specs"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logger.info("Starting LMSAuto...")
    
    try:
        # Initialize shared context
        context = SharedContext()
        
        # Profile system hardware if requested
        if args.profile_system:
            logger.info("Profiling system hardware...")
            profiler = SystemProfiler(context)
            profiler.profile_system()
            
            # Display hardware specs
            specs = context.get_hardware_specs()
            if specs:
                print("\n=== System Hardware Specs ===")
                for key, value in specs.items():
                    print(f"{key.replace('_', ' ').title()}: {value}")
                print("==============================\n")
        
        # Initialize and run model scanner
        logger.info("Initializing model scanner...")
        scanner = ModelScanner(context)
        
        # Register available scanners
        lm_studio_scanner = LMStudioScanner()
        scanner.register_scanner(lm_studio_scanner)
        
        # Scan for models
        logger.info("Scanning for models...")
        scanner.scan()
        
        # Display discovered models
        models = context.get_all_models()
        if models:
            print(f"\n=== Discovered Models ({len(models)}) ===")
            for model_key, model_info in models.items():
                print(f"• {model_info.name} ({model_info.platform})")
                print(f"  Path: {model_info.path}")
            print("=====================================\n")
        else:
            print("\nNo models found. Make sure LM Studio is running with models loaded.")
            logger.warning("No models discovered. Check LM Studio installation and model availability.")
        
        if args.scan_only:
            logger.info("Scan-only mode complete.")
            return
        
        # TODO: Implement HuggingFace integration and config generation
        logger.info("Configuration generation not yet implemented.")
        print("Configuration generation will be added in future updates.")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)
    
    logger.info("LMSAuto completed successfully.")

if __name__ == "__main__":
    main()