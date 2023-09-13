import torch
pth_file_path = "weights_011.pth"

checkpoint = torch.load(pth_file_path)

model_state_dict = checkpoint['model_state_dict']

for param_name, param_value in model_state_dict.items():
    print(f"Parameter name: {param_name}")
    print(f"Parameter value:\n{param_value}")
