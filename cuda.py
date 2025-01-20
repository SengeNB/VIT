import torch
import pynvml

def get_gpu_info():
    if torch.cuda.is_available():
        print(f"检测到 {torch.cuda.device_count()} 个 GPU")
        
        pynvml.nvmlInit()
        
        for i in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = torch.cuda.get_device_name(i)
            
            total_memory = info.total / 1024**2  # Convert bytes to MB
            free_memory = info.free / 1024**2   # Convert bytes to MB
            used_memory = info.used / 1024**2   # Convert bytes to MB
            
            print(f"GPU {i}: {name}")
            print(f"  总内存: {total_memory:.2f} MB")
            print(f"  可用内存: {free_memory:.2f} MB")
            print(f"  已用内存: {used_memory:.2f} MB")
        
        pynvml.nvmlShutdown()
    else:
        print("未检测到 GPU")

if __name__ == "__main__":
    get_gpu_info()
