import torch


tensor = torch.rand(1000,1000)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tensor.to(device)

torch.linalg.svd(tensor)

print(torch.cuda.is_available())
print(torch.cuda.get_arch_list())


