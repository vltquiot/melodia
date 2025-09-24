import torch, bitsandbytes as bnb
print("Test des du montage GPU")
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("FP16 supported:", torch.cuda.is_available())
print("BF16 supported:", torch.cuda.is_bf16_supported() if hasattr(torch, "is_bf16_supported") else False)
print("bitsandbytes ok")
