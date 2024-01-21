import torch

# Check if CUDA (GPU support) is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# If CUDA is available, print the GPU details
if cuda_available:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Properties: {torch.cuda.get_device_properties(0)}")

