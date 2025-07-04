"""Classes for fetching and applying model settings."""

import logging
# import json  # Added for potential JSON parsing #type: ignore
from typing import Dict, Optional, Any
# Pylance might report 'partially unknown type' for hf_hub_download.
# This is often an environment/type-stub issue. Ensure
# 'huggingface_hub[typing]' is installed.
from huggingface_hub import HfApi, hf_hub_download  # type: ignore
from huggingface_hub.errors import (RepositoryNotFoundError,
                                     EntryNotFoundError)
from requests.exceptions import RequestException  # For network errors

logger = logging.getLogger(__name__)


class SettingsFetcher:
    """Fetches optimal model settings from Hugging Face."""

    def __init__(self):
        """Initialize Hugging Face API client."""
        # self.hf_api is currently unused in find_optimal_settings
        self.hf_api = HfApi()
        self.logger = logging.getLogger(__name__)  # Use instance logger

    def find_optimal_settings(self, model_name_or_path: str) -> \
            Optional[Dict[str, Any]]:
        """Search Hugging Face for optimal settings for a given model.

        This might involve looking for specific config files (e.g.,
        config.json, generation_config.json) or parsing model card info.

        Args:
            model_name_or_path: The Hugging Face model ID (e.g.,
                'meta-llama/Llama-2-7b-chat-hf') or a local path.

        Returns:
            Dictionary containing optimal settings if found, None otherwise.
        """
        self.logger.info(
            f"Searching for optimal settings for model: {model_name_or_path}"
        )

        # Attempt 1: Look for generation_config.json
        try:
            config_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename="generation_config.json"
            )
            self.logger.info(f"Found generation_config.json at {config_path}")
            # TODO: Implement actual loading and parsing of
            # generation_config.json
            # Example (replace with actual logic):
            # with open(config_path, 'r') as f:
            #     gen_config = json.load(f)
            #     # Extract relevant settings like temperature, top_p,
            #     # max_new_tokens etc.
            #     return {"temperature": gen_config.get("temperature", 0.7),
            #             "max_new_tokens": gen_config.get(
            #                 "max_length", 512)}  # Example keys
            return {"temperature": 0.7, "max_new_tokens": 512}  # Placeholder
        except (RepositoryNotFoundError, EntryNotFoundError):
            self.logger.info(
                f"No generation_config.json found for {model_name_or_path}."
            )
        except RequestException as e:
            self.logger.warning(
                f"Network error downloading generation_config.json for "
                f"{model_name_or_path}: {e}"
            )
        except Exception as e:  # Catch other potential errors during
            # download/parse
            self.logger.error(
                f"Error processing generation_config.json for "
                f"{model_name_or_path}: {e}",
                exc_info=True
            )

        # Attempt 2: Look for config.json
        try:
            config_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename="config.json"
            )
            self.logger.info(f"Found config.json at {config_path}")
            # TODO: Implement actual loading and parsing of config.json
            # Example (replace with actual logic):
            # with open(config_path, 'r') as f:
            #     main_config = json.load(f)
            #     # Extract relevant settings like context_length (might be
            #     # under model-specific keys)
            #     # This often contains architectural details rather than
            #     # generation params.
            #     return {"context_length": main_config.get(
            #         "max_position_embeddings", 4096)}  # Example key
            return {"context_length": 4096}  # Placeholder
        except (RepositoryNotFoundError, EntryNotFoundError):
            self.logger.info(
                f"No config.json found for {model_name_or_path}."
            )
        except RequestException as e:
            self.logger.warning(
                f"Network error downloading config.json for "
                f"{model_name_or_path}: {e}"
            )
        except Exception as e:  # Catch other potential errors during
            # download/parse
            self.logger.error(
                f"Error processing config.json for {model_name_or_path}: {e}",
                exc_info=True
            )

        # Attempt 3: Parse model card (README.md) - More complex,
        # requires parsing Markdown
        try:
            readme_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename="README.md"
            )
            self.logger.info(
                f"Found README.md at {readme_path}. Parsing not yet "
                f"implemented."
            )
            # TODO: Implement README parsing logic (e.g., using regex or a
            # Markdown parser)
            # Look for sections describing recommended settings or
            # parameters.
        except (RepositoryNotFoundError, EntryNotFoundError):
            self.logger.info(
                f"No README.md found for {model_name_or_path}."
            )
        except RequestException as e:
            self.logger.warning(
                f"Network error downloading README.md for "
                f"{model_name_or_path}: {e}"
            )
        except Exception as e:  # Catch other potential errors during
            # download/parse
            self.logger.error(
                f"Error processing README.md for {model_name_or_path}: {e}",
                exc_info=True
            )

        self.logger.warning(
            f"Could not find or parse optimal settings for model: "
            f"{model_name_or_path}"
        )
        return None


# Note: Applying settings might be better handled within the LMStudioAPI
# class in models.py, as it requires direct interaction with the LM Studio
# instance. This class focuses solely on *fetching* potential settings.
