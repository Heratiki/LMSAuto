# lmsauto/scanner/__init__.py

import abc
import logging
from pathlib import Path
from typing import List, Optional # Removed unused 'Dict'

# Direct import - removed the try/except block
from lmsauto.shared.context import SharedContext, ModelInfo

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlatformScanner(abc.ABC):
    """Abstract base class for platform-specific model scanners."""

    PLATFORM_NAME: str = "Unknown"

    @abc.abstractmethod
    def discover_models(self) -> List[ModelInfo]:
        """
        Discover models for the specific platform.

        Returns:
            List[ModelInfo]: A list of discovered model information objects.
        """
        pass

class LMStudioScanner(PlatformScanner):
    """Scanner implementation for LM Studio models."""

    PLATFORM_NAME = "LM Studio"

    def __init__(self, base_path: Optional[str] = None):
        """
        Initializes the LMStudioScanner.

        Args:
            base_path (Optional[str]): The base directory where LM Studio models are stored.
                                       Defaults to ~/.cache/lm-studio/models if None.
        """
        if base_path:
            self.models_base_path = Path(base_path)
        else:
            # Default path for LM Studio models (adjust if necessary for different OS)
            self.models_base_path = Path.home() / ".cache" / "lm-studio" / "models"
        logger.info(f"LM Studio scanner initialized. Model path: {self.models_base_path}")

    def discover_models(self) -> List[ModelInfo]:
        """
        Discovers models stored by LM Studio.

        Assumes models are stored in subdirectories under the models_base_path.
        The subdirectory structure might be like: publisher/repo/model.gguf

        Returns:
            List[ModelInfo]: A list of discovered LM Studio models.
        """
        discovered_models: List[ModelInfo] = []
        if not self.models_base_path.is_dir():
            logger.warning(f"LM Studio model directory not found or is not a directory: {self.models_base_path}")
            return discovered_models

        try:
            # LM Studio often uses a nested structure like publisher/repo/model.gguf
            # We iterate through potential model files (.gguf is common)
            logger.debug(f"Scanning directory: {self.models_base_path}")
            for model_path in self.models_base_path.rglob('*.gguf'):
                logger.debug(f"Found potential model file: {model_path}")
                if model_path.is_file():
                    try:
                        # Attempt to derive a meaningful name from the path parts
                        # e.g., publisher/repo/model.gguf -> publisher/repo
                        try:
                            relative_path = model_path.relative_to(self.models_base_path)
                            logger.debug(f"Relative path: {relative_path}")
                        except Exception as e:
                            logger.error(f"Error deriving relative path for {model_path}: {e}")
                            continue
                        model_name = "/".join(relative_path.parts[:-1]) # Exclude filename
                        if not model_name: # Handle models directly in the root
                            model_name = relative_path.stem

                        model_info = ModelInfo(
                            name=model_name,
                            path=str(model_path.resolve()),
                            platform=self.PLATFORM_NAME,
                            # Add other relevant metadata if discoverable
                        )
                        discovered_models.append(model_info)
                        logger.info(f"Discovered LM Studio model: {model_name} at {model_path}")
                    except Exception as e:
                        logger.error(f"Error processing potential LM Studio model file {model_path}: {e}")

        except Exception as e:
            logger.error(f"Error scanning LM Studio model directory {self.models_base_path}: {e}")

        return discovered_models

class ModelScanner:
    """
    Discovers models across various platforms using registered scanners.
    """

    def __init__(self, context: SharedContext):
        """
        Initializes the ModelScanner.

        Args:
            context (SharedContext): The shared context object to store discovered models.
        """
        self.context: SharedContext = context
        self._scanners: List[PlatformScanner] = []
        logger.info("ModelScanner initialized.")

    def register_scanner(self, scanner_instance: PlatformScanner) -> None:
        """
        Registers a platform-specific scanner instance.

        Args:
            scanner_instance (PlatformScanner): An instance of a PlatformScanner subclass.
        """
        self._scanners.append(scanner_instance)
        logger.info(f"Registered scanner for platform: {scanner_instance.PLATFORM_NAME}")

    def scan(self) -> None:
        """
        Executes all registered scanners to discover models and updates the shared context.
        """
        logger.info("Starting model scan across registered platforms...")
        all_discovered_models: List[ModelInfo] = []
        for scanner in self._scanners:
            try:
                logger.info(f"Running scanner for {scanner.PLATFORM_NAME}...")
                discovered = scanner.discover_models()
                all_discovered_models.extend(discovered)
                logger.info(f"Scanner for {scanner.PLATFORM_NAME} found {len(discovered)} models.")
            except Exception as e:
                logger.error(f"Error running scanner for {scanner.PLATFORM_NAME}: {e}", exc_info=True)

        logger.info(f"Total models discovered across all platforms: {len(all_discovered_models)}")

        # Update shared context
        for model_info in all_discovered_models:
            try:
                self.context.add_model(model_info)
            except Exception as e:
                logger.error(f"Error adding model '{model_info.name}' to context: {e}")

        logger.info("Model scan complete. Shared context updated.")

# Example Usage (Optional - for testing or direct script execution)
if __name__ == '__main__':
    # This block will only execute when the script is run directly
    # It's useful for testing the scanner module independently

    # Create a dummy SharedContext for testing
    test_context = SharedContext()

    # Initialize the main scanner
    main_scanner = ModelScanner(context=test_context)

    # Register the LM Studio scanner (add others as they are implemented)
    lm_studio_scanner = LMStudioScanner()
    main_scanner.register_scanner(lm_studio_scanner)

    # TODO: Register OllamaScanner() once implemented
    # TODO: Register VLLMScanner() once implemented

    # Run the scan
    main_scanner.scan()

    # Print discovered models from the context
    print("\nDiscovered models in context:")
    # Use the get_all_models method for safe access
    models_in_context = test_context.get_all_models()
    if models_in_context:
        for model_key, model_info in models_in_context.items():
            # Accessing attributes should now work correctly
            print(f"- Key: {model_key}, Name: {model_info.name}, Platform: {model_info.platform}, Path: {model_info.path}")
    else:
        print("No models found.")