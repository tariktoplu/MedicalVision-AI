import torch

cuda_available = torch.cuda.is_available()
if cuda_available:
    print("CUDA kullanılabilir.")
else:
    print("CUDA kullanılamıyor, CPU kullanılacak.")