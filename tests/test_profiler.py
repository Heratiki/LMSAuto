from typing import Dict, Generator
import pytest
from unittest.mock import Mock, patch
from typing import cast

from lmsauto.profiler import SystemProfiler, BYTES_TO_GB
from lmsauto.shared.context import SharedContext

@pytest.fixture
def mock_shared_context() -> Mock:
    return Mock(spec=SharedContext)

@pytest.fixture
def mock_psutil() -> Generator[Mock, None, None]:
    with patch('lmsauto.profiler.psutil') as mock:
        yield mock

@pytest.fixture
def mock_torch() -> Generator[Mock, None, None]:
    with patch('lmsauto.profiler.torch') as mock:
        yield mock

@pytest.fixture
def mock_torch_not_available() -> Generator[None, None, None]:
    with patch('lmsauto.profiler._torch_available', False):
        yield

class TestSystemProfiler:
    
    def test_get_cpu_info_success(self, mock_shared_context: Mock, mock_psutil: Mock) -> None:
        # Arrange
        mock_psutil.cpu_count.side_effect = [4, 8]  # physical=4, logical=8
        profiler = SystemProfiler(mock_shared_context)
        
        # Act
        # pylint: disable=protected-access
        result = profiler._get_cpu_info()  # type: ignore[attr-defined]
        
        # Assert
        assert result == {
            'cpu_physical_cores': '4',
            'cpu_total_cores': '8'
        }
        mock_psutil.cpu_count.assert_any_call(logical=False)
        mock_psutil.cpu_count.assert_any_call(logical=True)

    def test_get_cpu_info_not_implemented(self, mock_shared_context: Mock, mock_psutil: Mock) -> None:
        # Arrange
        mock_psutil.cpu_count.side_effect = NotImplementedError()
        profiler = SystemProfiler(mock_shared_context)
        
        # Act
        # pylint: disable=protected-access
        result = profiler._get_cpu_info()  # type: ignore[attr-defined]
        
        # Assert
        assert result == {
            'cpu_physical_cores': 'Not Implemented',
            'cpu_total_cores': 'Not Implemented'
        }

    def test_get_cpu_info_error(self, mock_shared_context: Mock, mock_psutil: Mock) -> None:
        # Arrange
        mock_psutil.cpu_count.side_effect = Exception("Test error")
        profiler = SystemProfiler(mock_shared_context)
        
        # Act
        # pylint: disable=protected-access
        result = profiler._get_cpu_info()  # type: ignore[attr-defined]
        
        # Assert
        assert result == {
            'cpu_physical_cores': 'Error',
            'cpu_total_cores': 'Error'
        }

    def test_get_ram_info_success(self, mock_shared_context: Mock, mock_psutil: Mock) -> None:
        # Arrange
        mock_vmem = Mock()
        mock_vmem.total = 16 * BYTES_TO_GB  # 16 GB total
        mock_vmem.available = 8 * BYTES_TO_GB  # 8 GB available
        mock_psutil.virtual_memory.return_value = mock_vmem
        profiler = SystemProfiler(mock_shared_context)
        
        # Act
        # pylint: disable=protected-access
        result = profiler._get_ram_info()  # type: ignore[attr-defined]
        
        # Assert
        assert result == {
            'ram_total_gb': '16.00',
            'ram_available_gb': '8.00'
        }
        mock_psutil.virtual_memory.assert_called_once()

    def test_get_ram_info_error(self, mock_shared_context: Mock, mock_psutil: Mock) -> None:
        # Arrange
        mock_psutil.virtual_memory.side_effect = Exception("Test error")
        profiler = SystemProfiler(mock_shared_context)
        
        # Act
        # pylint: disable=protected-access
        result = profiler._get_ram_info()  # type: ignore[attr-defined]
        
        # Assert
        assert result == {
            'ram_total_gb': 'Error',
            'ram_available_gb': 'Error'
        }

    @pytest.mark.parametrize("cuda_available,device_count,expected", [
        (True, 2, {
            'gpu_available': 'Yes (CUDA)',
            'gpu_details': '2 devices: GPU0 (8.00 GB); GPU1 (8.00 GB)',
            'gpu_vram_total_gb': '16.00'
        }),
        (False, 0, {
            'gpu_available': 'No (CUDA)',
            'gpu_details': 'N/A',
            'gpu_vram_total_gb': 'N/A'
        })
    ])
    def test_get_gpu_info_with_torch(
        self, 
        mock_shared_context: Mock, 
        mock_torch: Mock, 
        cuda_available: bool, 
        device_count: int, 
        expected: Dict[str, str]
    ) -> None:
        # Arrange
        mock_torch.cuda.is_available.return_value = cuda_available
        mock_torch.cuda.device_count.return_value = device_count
        if cuda_available:
            mock_torch.cuda.get_device_name.side_effect = ['GPU0', 'GPU1']
            mock_properties = Mock()
            mock_properties.total_memory = 8 * BYTES_TO_GB
            mock_torch.cuda.get_device_properties.return_value = mock_properties
        profiler = SystemProfiler(mock_shared_context)
        
        # Act
        # pylint: disable=protected-access
        result = profiler._get_gpu_info()  # type: ignore[attr-defined]
        
        # Assert
        assert result == expected

    def test_get_gpu_info_no_torch(self, mock_shared_context: Mock, mock_torch_not_available: None) -> None:
        # Arrange
        profiler = SystemProfiler(mock_shared_context)
        
        # Act
        # pylint: disable=protected-access
        result = profiler._get_gpu_info()  # type: ignore[attr-defined]
        
        # Assert
        assert result == {
            'gpu_available': 'Unknown (torch not found)',
            'gpu_details': 'N/A',
            'gpu_vram_total_gb': 'N/A'
        }

    def test_get_gpu_info_cuda_error(self, mock_shared_context: Mock, mock_torch: Mock) -> None:
        # Arrange
        mock_torch.cuda.is_available.side_effect = Exception("CUDA error")
        profiler = SystemProfiler(mock_shared_context)
        
        # Act
        # pylint: disable=protected-access
        result = profiler._get_gpu_info()  # type: ignore[attr-defined]
        
        # Assert
        assert result == {
            'gpu_available': 'Error (torch check failed)',
            'gpu_details': 'N/A',
            'gpu_vram_total_gb': 'N/A'
        }

    def test_get_python_info_success(self, mock_shared_context: Mock) -> None:
        # Arrange
        profiler = SystemProfiler(mock_shared_context)
        
        # Act
        # pylint: disable=protected-access
        result = profiler._get_python_info()  # type: ignore[attr-defined]
        
        # Assert
        assert 'python_version' in result
        assert 'python_executable' in result
        assert isinstance(result['python_version'], str)
        assert isinstance(result['python_executable'], str)
        assert result['python_version'] != 'Error'
        assert result['python_executable'] != 'Error'

    def test_profile_system_success(self, mock_shared_context: Mock, mock_psutil: Mock, mock_torch: Mock) -> None:
        # Arrange
        # Setup CPU mocks
        mock_psutil.cpu_count.side_effect = [4, 8]
        
        # Setup RAM mocks
        mock_vmem = Mock()
        mock_vmem.total = 16 * BYTES_TO_GB
        mock_vmem.available = 8 * BYTES_TO_GB
        mock_psutil.virtual_memory.return_value = mock_vmem
        
        # Setup GPU mocks
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = 'Test GPU'
        mock_properties = Mock()
        mock_properties.total_memory = 8 * BYTES_TO_GB
        mock_torch.cuda.get_device_properties.return_value = mock_properties
        
        profiler = SystemProfiler(mock_shared_context)
        
        # Act
        profiler.profile_system()
        
        # Assert
        # Verify SharedContext.set_hardware_specs was called with expected specs
        mock_shared_context.set_hardware_specs.assert_called_once()
        specs = cast(Dict[str, str], mock_shared_context.set_hardware_specs.call_args[0][0])
        assert isinstance(specs, dict)
        assert specs['cpu_physical_cores'] == '4'
        assert specs['cpu_total_cores'] == '8'
        assert specs['ram_total_gb'] == '16.00'
        assert specs['ram_available_gb'] == '8.00'
        assert specs['gpu_available'] == 'Yes (CUDA)'
        assert 'Test GPU' in specs['gpu_details']
        assert specs['gpu_vram_total_gb'] == '8.00'
        assert 'python_version' in specs
        assert 'python_executable' in specs

    def test_profile_system_handle_context_error(self, mock_shared_context: Mock, mock_psutil: Mock, mock_torch: Mock) -> None:
        # Arrange
        mock_shared_context.set_hardware_specs.side_effect = Exception("Context error")
        profiler = SystemProfiler(mock_shared_context)
        
        # Act & Assert
        # Should not raise exception
        profiler.profile_system()
        mock_shared_context.set_hardware_specs.assert_called_once()