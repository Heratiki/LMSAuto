# lmsauto/profiler/__init__.py
import logging
import sys
from typing import Dict, Optional, TYPE_CHECKING

# Third-party imports - ensure psutil is installed
try:
    import psutil
except ImportError:
    print("Error: psutil library not found. Please install it: pip install psutil")
    sys.exit(1)

# Attempt optional import for GPU detection
_torch_available = False
try:
    import torch
    _torch_available = True
except ImportError:
    pass # Keep _torch_available as False

# Conditionally import torch for type checking only
if TYPE_CHECKING:
    import torch # Allow type checker to see torch types

# Local application imports (adjust path based on project structure)
# Assuming SharedContext is in lmsauto.shared.context
try:
    from ..shared.context import SharedContext
except ImportError:
    # Fallback for potential execution context issues (e.g., running script directly)
    from lmsauto.shared.context import SharedContext


# Configure basic logging
# Consider moving logging configuration to a central place (e.g., cli.py or __main__)
# For now, basic config here for module-level testing/use.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BYTES_TO_GB = 1024**3

class SystemProfiler:
    """
    Analyzes system hardware resources (CPU, RAM, GPU) and stores
    the information in the shared context.

    Uses psutil for CPU and RAM information. Attempts basic GPU detection,
    currently checking for CUDA availability via PyTorch if installed.
    """

    def __init__(self, shared_context: SharedContext):
        """
        Initializes the SystemProfiler.

        Args:
            shared_context: The shared context object to store results.

        Raises:
            TypeError: If shared_context is not an instance of SharedContext.
        """
        # Type is enforced by the type hint, isinstance check is redundant
        self.context = shared_context
        logger.info("SystemProfiler initialized.")

    def _get_cpu_info(self) -> Dict[str, str]:
        """Gathers CPU information using psutil."""
        specs: Dict[str, str] = {}
        try:
            physical_cores: Optional[int] = psutil.cpu_count(logical=False)
            total_cores: Optional[int] = psutil.cpu_count(logical=True)
            specs['cpu_physical_cores'] = str(physical_cores) if physical_cores is not None else 'N/A'
            specs['cpu_total_cores'] = str(total_cores) if total_cores is not None else 'N/A'
            # Future enhancement: Add CPU frequency, model name if easily available
            logger.info(f"CPU Info: Physical Cores={specs['cpu_physical_cores']}, Total Cores={specs['cpu_total_cores']}")
        except NotImplementedError:
             logger.warning("CPU core count detection not implemented for this platform by psutil.")
             specs['cpu_physical_cores'] = 'Not Implemented'
             specs['cpu_total_cores'] = 'Not Implemented'
        except Exception as e:
            logger.error(f"Error getting CPU info: {e}", exc_info=True)
            specs['cpu_physical_cores'] = 'Error'
            specs['cpu_total_cores'] = 'Error'
        return specs

    def _get_ram_info(self) -> Dict[str, str]:
        """Gathers RAM information using psutil."""
        specs: Dict[str, str] = {}
        try:
            vmem = psutil.virtual_memory()
            total_gb = vmem.total / BYTES_TO_GB
            available_gb = vmem.available / BYTES_TO_GB
            specs['ram_total_gb'] = f"{total_gb:.2f}"
            specs['ram_available_gb'] = f"{available_gb:.2f}" # Available might fluctuate
            logger.info(f"RAM Info: Total={specs['ram_total_gb']} GB, Available={specs['ram_available_gb']} GB")
        except Exception as e:
            logger.error(f"Error getting RAM info: {e}", exc_info=True)
            specs['ram_total_gb'] = 'Error'
            specs['ram_available_gb'] = 'Error'
        return specs

    def _get_gpu_info(self) -> Dict[str, str]:
        """
        Attempts basic GPU detection. Currently checks for CUDA via torch.
        Future: Could integrate GPUtil or platform-specific tools.
        """
        specs: Dict[str, str] = {}
        gpu_available = "Unknown"
        gpu_details = "N/A"
        gpu_vram_gb = "N/A" # Placeholder for VRAM

        if _torch_available:
            try:
                # Type checker now knows about torch types due to conditional import
                if torch.cuda.is_available():
                    gpu_available = "Yes (CUDA)"
                    device_count: int = torch.cuda.device_count()
                    details_list: list[str] = []
                    total_vram: float = 0.0
                    for i in range(device_count):
                        device_name: str = torch.cuda.get_device_name(i)
                        try:
                            # Access properties directly with type annotations
                            properties = torch.cuda.get_device_properties(i)  # type: ignore[attr-defined]
                            # Explicitly convert total_memory to float to ensure type safety
                            memory_bytes: float = float(properties.total_memory)  # type: ignore[attr-defined]
                            vram: float = memory_bytes / BYTES_TO_GB
                            # Use explicit assignment to maintain float type
                            total_vram = float(total_vram + vram)
                            details_list.append(f"{device_name} ({vram:.2f} GB)")
                        except Exception as mem_e:
                            logger.warning(f"Could not get VRAM for GPU {i}: {mem_e}")
                            details_list.append(f"{device_name} (VRAM N/A)")

                    gpu_details = f"{device_count} devices: {'; '.join(details_list)}"
                    gpu_vram_gb = f"{total_vram:.2f}" if total_vram > 0 else "N/A"

                else:
                    gpu_available = "No (CUDA)"
                    # Check for other backends like MPS (Apple Silicon) if needed
                    # if torch.backends.mps.is_available():  # type: ignore
                    #     gpu_available = "Yes (MPS)"
                    #     gpu_details = "Apple Silicon GPU" # Basic detail
            except Exception as e:
                logger.warning(f"Could not check GPU via torch: {e}", exc_info=True)
                gpu_available = "Error (torch check failed)"
        else:
            gpu_available = "Unknown (torch not found)"

        specs['gpu_available'] = gpu_available
        specs['gpu_details'] = gpu_details
        specs['gpu_vram_total_gb'] = gpu_vram_gb # Add VRAM info
        logger.info(f"GPU Info: Available={specs['gpu_available']}, Details={specs['gpu_details']}, Total VRAM={specs['gpu_vram_total_gb']} GB")
        return specs

    def _get_python_info(self) -> Dict[str, str]:
        """Gathers Python runtime information."""
        specs: Dict[str, str] = {}
        try:
            specs['python_version'] = sys.version.split()[0]
            specs['python_executable'] = sys.executable
            logger.info(f"Python Version: {specs['python_version']}")
        except Exception as e:
            logger.error(f"Error getting Python info: {e}", exc_info=True)
            specs['python_version'] = 'Error'
            specs['python_executable'] = 'Error'
        return specs

    def profile_system(self) -> None:
        """
        Performs system profiling (CPU, RAM, GPU, Python) and updates
        the shared context's hardware_specs dictionary.
        """
        logger.info("Starting system profiling...")
        # Use a temporary dict to gather all specs before updating context
        current_specs: Dict[str, str] = {}

        current_specs.update(self._get_cpu_info())
        current_specs.update(self._get_ram_info())
        current_specs.update(self._get_gpu_info())
        current_specs.update(self._get_python_info())

        # Update shared context thread-safely
        try:
            # Use the thread-safe setter method from SharedContext
            self.context.set_hardware_specs(current_specs)
            logger.info("System profiling complete. Shared context updated.")
            logger.debug(f"Updated Hardware Specs in Context: {current_specs}")
        except Exception as e:
             logger.error(f"Failed to update shared context with hardware specs: {e}", exc_info=True)


# --- Example Usage ---
# This block allows testing the profiler directly.
# In the main application, instantiate SharedContext and pass it.
if __name__ == '__main__':
    print("--- Running SystemProfiler Standalone Test ---")
    # Create a mock SharedContext instance for testing
    try:
        mock_context = SharedContext()
        profiler = SystemProfiler(mock_context)
        profiler.profile_system()

        print("\n--- Hardware Specs Collected ---")
        # Access context safely for printing results
        # Use the thread-safe getter method
        specs_copy = mock_context.get_hardware_specs()

        if specs_copy:
            for key, value in specs_copy.items():
                print(f"- {key.replace('_', ' ').title()}: {value}")
        else:
            print("No hardware specs were collected (check logs for errors).")
        print("---------------------------------")

    except Exception as main_e:
        print(f"\nAn error occurred during the standalone test: {main_e}")
        logger.exception("Standalone test failed")

    print("--- Standalone Test Complete ---")