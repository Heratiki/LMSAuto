"""Main entry point for the LMSAuto application."""

import argparse
import logging

from .utils import setup_logging
from .models import LMStudioAPI, ModelProfileManager
from .settings import SettingsFetcher

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """Main function to run the LMSAuto tool."""
    parser = argparse.ArgumentParser(description="LM Studio Autonomous Model Settings Optimizer")
    parser.add_argument(
        "--lmstudio-url",
        default="http://localhost:1234",
        help="URL of the LM Studio API endpoint."
    )
    parser.add_argument(
        "--profiles-dir",
        default="profiles",
        help="Directory to store model configuration profiles."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    # Add more arguments as needed (e.g., specific model to optimize, action to perform)

    args = parser.parse_args()

    # Re-setup logging with potentially different level
    setup_logging(args.log_level)
    logger.info("Starting LMSAuto...")
    logger.debug(f"Arguments: {args}")

    # Initialize components
    lm_studio = LMStudioAPI(api_url=args.lmstudio_url)
    profile_manager = ModelProfileManager(profiles_dir=args.profiles_dir)
    settings_fetcher = SettingsFetcher()

    # --- Core Logic ---
    # 1. Discover models from LM Studio
    try:
        available_models = lm_studio.discover_models()
        if not available_models:
            logger.warning("No models discovered via LM Studio API. Exiting.")
            # TODO: Add Rich UI feedback here
            return
        logger.info(f"Discovered models: {[m['name'] for m in available_models]}")
    except Exception as e:
        logger.error(f"Failed to discover models from LM Studio: {e}")
        # TODO: Add Rich UI feedback here
        return

    # 2. For each model, try to find optimal settings and save profile
    for model_info in available_models:
        model_name = model_info.get("name")
        model_path = model_info.get("path") # LM Studio might provide a path or identifier

        if not model_name:
            logger.warning(f"Skipping model with missing name: {model_info}")
            continue

        logger.info(f"Processing model: {model_name}")

        # Use model name or path for searching Hugging Face
        # Prioritize path if available, otherwise use name (might need mapping)
        hf_identifier = model_path if model_path else model_name
        # TODO: Refine how hf_identifier is determined (might need user input or mapping logic)

        optimal_settings = settings_fetcher.find_optimal_settings(hf_identifier)

        if optimal_settings:
            logger.info(f"Found potential optimal settings for {model_name}: {optimal_settings}")
            # TODO: Validate settings format before saving?
            profile_manager.save_profile(model_name, optimal_settings)
        else:
            logger.warning(f"Could not find optimal settings for {model_name}")

    # 3. TODO: Implement Rich UI for user interaction (selecting profiles, applying settings)
    logger.info("Settings search complete. Profiles saved.")
    logger.info("Next steps: Implement Rich UI and settings application logic.")

    # Example: Load and print a profile
    # test_model = "ExampleModel" # Replace with an actual model name if a profile was saved
    # loaded_profile = profile_manager.load_profile(test_model)
    # if loaded_profile:
    #     logger.info(f"Loaded profile for {test_model}: {loaded_profile}")


if __name__ == "__main__":
    main()