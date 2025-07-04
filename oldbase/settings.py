"""Classes for fetching and applying model settings."""

import logging
import json
import re
from typing import Dict, Optional, Any
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import (RepositoryNotFoundError,
                                     EntryNotFoundError)
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


class SettingsFetcher:
    """Fetches optimal model settings from Hugging Face."""

    def __init__(self):
        """Initialize Hugging Face API client."""
        self.hf_api = HfApi()
        self.logger = logging.getLogger(__name__)

    def _map_to_hf_repo(self, model_name: str) -> Optional[str]:
        """Map LM Studio model name to Hugging Face repository name.
        
        Args:
            model_name: LM Studio model name (e.g., 'qwen2.5-coder-7b-instruct')
            
        Returns:
            Hugging Face repository name or None if no mapping found
        """
        # Common mapping patterns
        mappings = {
            # Qwen models
            r'qwen2\.5-coder-(\d+)b-instruct': r'Qwen/Qwen2.5-Coder-\1B-Instruct',
            r'qwen2-(\d+\.?\d*)b-instruct': r'Qwen/Qwen2-\1B-Instruct',
            
            # Gemma models
            r'gemma-3-(\d+)b-it': r'google/gemma-2-\1b-it',
            
            # StarCoder models
            r'starcoder2-(\d+)b(?:-instruct)?': r'bigcode/starcoder2-\1b',
            
            # DeepSeek models
            r'deepseek-coder-(\d+\.?\d*)b-(\w+)': r'deepseek-ai/deepseek-coder-\1b-\2',
            r'deepseek-r1-distill-(\w+)-(\d+\.?\d*)b': r'deepseek-ai/deepseek-r1-distill-\1-\2b',
            
            # CodeLlama models
            r'codellama-(\d+)b-instruct': r'codellama/CodeLlama-\1b-Instruct-hf',
            
            # Stable Code models
            r'stable-code-instruct-(\d+)b': r'stabilityai/stable-code-instruct-\1b',
            
            # OpenAI Community models
            r'openai-community_-_(.+)': r'openai-community/\1',
            
            # Embedding models
            r'text-embedding-nomic-embed-text-v(\d+\.?\d*)': r'nomic-ai/nomic-embed-text-v\1',
        }
        
        for pattern, replacement in mappings.items():
            match = re.match(pattern, model_name, re.IGNORECASE)
            if match:
                repo_name = re.sub(pattern, replacement, model_name, flags=re.IGNORECASE)
                self.logger.info(f"Mapped '{model_name}' to '{repo_name}'")
                return repo_name
        
        # If no mapping found, try the original name
        self.logger.warning(f"No mapping found for '{model_name}', trying original name")
        return model_name

    def find_optimal_settings(self, model_name_or_path: str) -> \
            Optional[Dict[str, Any]]:
        """Search Hugging Face for optimal settings for a given model.

        Args:
            model_name_or_path: The LM Studio model name or HF repo ID

        Returns:
            Dictionary containing optimal settings if found, None otherwise.
        """
        self.logger.info(
            f"Searching for optimal settings for model: {model_name_or_path}"
        )

        # Map LM Studio model name to HF repository name
        hf_repo_name = self._map_to_hf_repo(model_name_or_path)
        if not hf_repo_name:
            return None

        settings = {}

        # Attempt 1: Look for generation_config.json
        try:
            config_path = hf_hub_download(
                repo_id=hf_repo_name,
                filename="generation_config.json"
            )
            self.logger.info(f"Found generation_config.json at {config_path}")
            
            with open(config_path, 'r') as f:
                gen_config = json.load(f)
                # Extract relevant generation settings
                if "temperature" in gen_config:
                    settings["temperature"] = gen_config["temperature"]
                if "top_p" in gen_config:
                    settings["top_p"] = gen_config["top_p"]
                if "top_k" in gen_config:
                    settings["top_k"] = gen_config["top_k"]
                if "max_new_tokens" in gen_config:
                    settings["max_new_tokens"] = gen_config["max_new_tokens"]
                elif "max_length" in gen_config:
                    settings["max_new_tokens"] = gen_config["max_length"]
                if "repetition_penalty" in gen_config:
                    settings["repetition_penalty"] = gen_config["repetition_penalty"]
                
                self.logger.info(f"Extracted generation settings: {settings}")
                
        except (RepositoryNotFoundError, EntryNotFoundError):
            self.logger.info(
                f"No generation_config.json found for {hf_repo_name}."
            )
        except RequestException as e:
            self.logger.warning(
                f"Network error downloading generation_config.json for "
                f"{hf_repo_name}: {e}"
            )
        except Exception as e:
            self.logger.error(
                f"Error processing generation_config.json for "
                f"{hf_repo_name}: {e}",
                exc_info=True
            )

        # Attempt 2: Look for config.json
        try:
            config_path = hf_hub_download(
                repo_id=hf_repo_name,
                filename="config.json"
            )
            self.logger.info(f"Found config.json at {config_path}")
            
            with open(config_path, 'r') as f:
                main_config = json.load(f)
                # Extract architectural settings
                if "max_position_embeddings" in main_config:
                    settings["max_context_length"] = main_config["max_position_embeddings"]
                elif "max_sequence_length" in main_config:
                    settings["max_context_length"] = main_config["max_sequence_length"]
                elif "seq_length" in main_config:
                    settings["max_context_length"] = main_config["seq_length"]
                
                # Extract model architecture info
                if "model_type" in main_config:
                    settings["model_type"] = main_config["model_type"]
                if "vocab_size" in main_config:
                    settings["vocab_size"] = main_config["vocab_size"]
                    
                self.logger.info(f"Extracted config settings: {dict(main_config)}")
                
        except (RepositoryNotFoundError, EntryNotFoundError):
            self.logger.info(
                f"No config.json found for {hf_repo_name}."
            )
        except RequestException as e:
            self.logger.warning(
                f"Network error downloading config.json for "
                f"{hf_repo_name}: {e}"
            )
        except Exception as e:
            self.logger.error(
                f"Error processing config.json for {hf_repo_name}: {e}",
                exc_info=True
            )

        # Return settings if we found any
        if settings:
            self.logger.info(f"Found optimal settings for {model_name_or_path}: {settings}")
            return settings
        
        self.logger.warning(
            f"Could not find or parse optimal settings for model: "
            f"{model_name_or_path}"
        )
        return None


# Note: Applying settings might be better handled within the LMStudioAPI
# class in models.py, as it requires direct interaction with the LM Studio
# instance. This class focuses solely on *fetching* potential settings.
