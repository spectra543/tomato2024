import torch
print(f"###############################\n## Cuda availability testing ##\n###############################\nGPU available: {torch.cuda.is_available()}\nTorch cuda device: {torch.cuda.device(0)} \nDevice name: {torch.cuda.get_device_name(0)}\n###############################\n##### Testing completed ##### \n###############################")


