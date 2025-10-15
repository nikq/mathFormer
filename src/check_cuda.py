import torch, sys, platform

print("Python:", sys.version)
print("Platform:", platform.platform())
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Built with CUDA:", torch.version.cuda)
print("Compiled w/ cuDNN:", getattr(torch.backends, 'cudnn', None) and torch.backends.cudnn.version())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f" - Device {i}: {torch.cuda.get_device_name(i)}")
    x = torch.rand(2,2).cuda()
    print("Tensor device:", x.device)
else:
    print("Reason (common): driver mismatch / package CPU build / environment missing cuda feature")
