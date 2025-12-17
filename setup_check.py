"""
Setup verification script for Llama 3 8B fine-tuning project.
Checks all dependencies, system configuration, and environment setup.
"""

import sys
import os
import platform
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}\n")

def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.END} {text}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.END} {text}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")

def print_info(text):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ{Colors.END} {text}")

# Required packages from requirements.txt
REQUIRED_PACKAGES = [
    'torch',
    'transformers',
    'datasets',
    'accelerate',
    'peft',
    'bitsandbytes',
    'gradio',
    'sentence_transformers',
    'PyPDF2'
]

def check_python_version():
    """Check Python version."""
    print_header("Python Version Check")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print_info(f"Python Version: {version_str}")
    
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version_str} is compatible (requires 3.8+)")
        return True
    else:
        print_error(f"Python {version_str} is not compatible (requires 3.8+)")
        return False

def check_packages():
    """Check if all required packages are installed."""
    print_header("Package Installation Check")
    missing_packages = []
    installed_packages = []
    
    for package in REQUIRED_PACKAGES:
        # Handle package name variations
        import_name = package.replace('-', '_').lower()
        if package == 'sentence-transformers':
            import_name = 'sentence_transformers'
        elif package == 'PyPDF2':
            import_name = 'PyPDF2'
        elif package == 'bitsandbytes':
            import_name = 'bitsandbytes'
        
        try:
            __import__(import_name)
            installed_packages.append(package)
            print_success(f"{package} is installed")
        except ImportError:
            missing_packages.append(package)
            print_error(f"{package} is NOT installed")
    
    if missing_packages:
        print_warning(f"\nMissing packages: {', '.join(missing_packages)}")
        print_info("Install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print_success("All required packages are installed!")
        return True

def get_package_version(package_name):
    """Get version of an installed package."""
    try:
        # Try Python 3.8+ importlib.metadata
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except ImportError:
        try:
            # Fallback for Python 3.7
            import importlib_metadata
            return importlib_metadata.version(package_name)
        except ImportError:
            try:
                # Fallback for older Python versions
                import pkg_resources
                return pkg_resources.get_distribution(package_name).version
            except:
                # Last resort: try importing the package and checking __version__
                try:
                    # Handle package name variations
                    import_name = package_name.replace('-', '_').lower()
                    if package_name == 'sentence-transformers':
                        import_name = 'sentence_transformers'
                    elif package_name == 'PyPDF2':
                        import_name = 'PyPDF2'
                    
                    module = __import__(import_name)
                    if hasattr(module, '__version__'):
                        return module.__version__
                except:
                    pass
                return "Unknown"

def check_pytorch_and_cuda():
    """Check PyTorch installation and CUDA availability."""
    print_header("PyTorch and CUDA Check")
    
    try:
        import torch
        pytorch_version = torch.__version__
        print_success(f"PyTorch is installed: {pytorch_version}")
    except ImportError:
        print_error("PyTorch is NOT installed")
        print_info("Install PyTorch with: pip install torch")
        return False
    
    # Check CUDA availability
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print_success("CUDA is available!")
            print_info(f"CUDA Version: {torch.version.cuda}")
            try:
                print_info(f"cuDNN Version: {torch.backends.cudnn.version()}")
            except:
                print_info("cuDNN Version: Not available")
            
            # GPU information
            gpu_count = torch.cuda.device_count()
            print_info(f"Number of GPUs: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_memory = gpu_props.total_memory / (1024**3)  # GB
                print_info(f"  GPU {i}: {gpu_name}")
                print_info(f"    Total Memory: {gpu_memory:.2f} GB")
                print_info(f"    Compute Capability: {gpu_props.major}.{gpu_props.minor}")
                
                # Current memory usage
                try:
                    current_memory = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                    cached_memory = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                    print_info(f"    Allocated Memory: {current_memory:.2f} GB")
                    print_info(f"    Cached Memory: {cached_memory:.2f} GB")
                except:
                    pass
        else:
            print_warning("CUDA is NOT available")
            print_info("Training will run on CPU (much slower)")
            print_info("To use CUDA, ensure you have:")
            print_info("  1. NVIDIA GPU with CUDA support")
            print_info("  2. CUDA toolkit installed")
            print_info("  3. PyTorch with CUDA support (install from pytorch.org)")
    except Exception as e:
        print_warning(f"Error checking CUDA: {e}")
    
    return True

def check_huggingface_token():
    """Check if Hugging Face token is configured."""
    print_header("Hugging Face Token Check")
    
    token_configured = False
    token_source = None
    
    # Check environment variable
    hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    if hf_token:
        token_configured = True
        token_source = "environment variable"
        # Mask the token for security
        masked_token = hf_token[:8] + "..." + hf_token[-4:] if len(hf_token) > 12 else "***"
        print_success(f"Hugging Face token found in {token_source}")
        print_info(f"Token: {masked_token}")
    
    # Check Hugging Face cache directory and login status
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        if user_info and 'name' in user_info:
            token_configured = True
            token_source = "Hugging Face cache (logged in)"
            print_success(f"Hugging Face token configured ({token_source})")
            print_info(f"Logged in as: {user_info.get('name', 'Unknown')}")
    except ImportError:
        # huggingface_hub not installed, skip this check
        pass
    except Exception:
        # Not logged in or token invalid
        pass
    
    # Check token file in Hugging Face cache
    hf_home = os.environ.get('HF_HOME') or os.path.expanduser('~/.cache/huggingface')
    token_file = os.path.join(hf_home, 'token')
    if os.path.exists(token_file):
        if not token_configured:
            token_configured = True
            token_source = "token file"
            print_success(f"Hugging Face token found in {token_source}")
    
    # Also check in .huggingface directory
    hf_home_alt = os.path.expanduser('~/.huggingface')
    token_file_alt = os.path.join(hf_home_alt, 'token')
    if os.path.exists(token_file_alt) and not token_configured:
        token_configured = True
        token_source = "token file (.huggingface)"
        print_success(f"Hugging Face token found in {token_source}")
    
    if not token_configured:
        print_error("Hugging Face token is NOT configured")
        print_warning("You need a Hugging Face token to download Llama 3 models")
        print_info("To configure:")
        print_info("  1. Get token from: https://huggingface.co/settings/tokens")
        print_info("  2. Set environment variable: export HUGGING_FACE_HUB_TOKEN=your_token")
        print_info("  3. Or login: huggingface-cli login")
        return False
    else:
        return True

def check_directories():
    """Check if all necessary directories exist."""
    print_header("Directory Structure Check")
    
    try:
        import config
        directories = {
            'Specs': config.SPECS_DIR,
            'Processed Data': config.PROCESSED_DATA_DIR,
            'Models': config.MODELS_DIR,
            'Output': config.OUTPUT_DIR,
        }
    except ImportError:
        print_error("Could not import config.py")
        return False
    
    all_exist = True
    for name, path in directories.items():
        if os.path.exists(path) and os.path.isdir(path):
            print_success(f"{name} directory exists: {path}")
        else:
            print_error(f"{name} directory does NOT exist: {path}")
            all_exist = False
            # Try to create it
            try:
                os.makedirs(path, exist_ok=True)
                print_success(f"Created {name} directory: {path}")
                all_exist = True
            except Exception as e:
                print_error(f"Failed to create {name} directory: {e}")
    
    if all_exist:
        print_success("All required directories exist!")
    
    return all_exist

def print_system_info():
    """Print system information."""
    print_header("System Information")
    
    # Operating system
    print_info(f"Operating System: {platform.system()} {platform.release()}")
    print_info(f"Architecture: {platform.machine()}")
    print_info(f"Processor: {platform.processor()}")
    
    # Python version
    print_info(f"Python Version: {sys.version.split()[0]}")
    print_info(f"Python Executable: {sys.executable}")
    
    # Package versions
    print_info("\nPackage Versions:")
    packages_to_check = ['torch', 'transformers', 'datasets', 'accelerate', 'peft']
    for pkg in packages_to_check:
        try:
            version = get_package_version(pkg)
            print_info(f"  {pkg}: {version}")
        except:
            print_info(f"  {pkg}: Not installed")
    
    # CUDA version (if available)
    try:
        import torch
        if torch.cuda.is_available():
            print_info(f"PyTorch CUDA Version: {torch.version.cuda}")
            try:
                print_info(f"cuDNN Version: {torch.backends.cudnn.version()}")
            except:
                print_info("cuDNN Version: Not available")
        else:
            print_info("CUDA: Not available")
    except ImportError:
        print_info("PyTorch: Not installed")
    except:
        pass

def main():
    """Main function to run all checks."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("=" * 60)
    print("Llama 3 8B Fine-tuning Setup Verification")
    print("=" * 60)
    print(f"{Colors.END}\n")
    
    results = {
        'Python Version': check_python_version(),
        'Packages': check_packages(),
        'PyTorch/CUDA': check_pytorch_and_cuda(),
        'Hugging Face Token': check_huggingface_token(),
        'Directories': check_directories(),
    }
    
    print_system_info()
    
    # Summary
    print_header("Setup Check Summary")
    all_passed = True
    for check_name, passed in results.items():
        if passed:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")
            all_passed = False
    
    print("\n")
    if all_passed:
        print(f"{Colors.BOLD}{Colors.GREEN}✓ All checks passed! Your environment is ready for fine-tuning.{Colors.END}\n")
        return 0
    else:
        print(f"{Colors.BOLD}{Colors.YELLOW}⚠ Some checks failed. Please fix the issues above before proceeding.{Colors.END}\n")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

