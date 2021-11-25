import numpy as np
from VAE_helper import VAE
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

Batch_size = 1
model = VAE()
model.load_state_dict(torch.load('saved_model/model_99.pth'))
transform = transforms.Compose([transforms.CenterCrop(128),
                                transforms.ToTensor()])
dataset = datasets.ImageFolder('imgs', transform=transform)
train_DataLoader = DataLoader(dataset, batch_size=Batch_size, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
fig, (ax1, ax2) = plt.subplots(1,2)
for batch_idx, img_data in enumerate(train_DataLoader):
    img_data = img_data[0].to(device)
    encoded_latent, z_mean, z_log_var, decoded_img = model(img_data)
    decoded_img = torch.squeeze(decoded_img)
    decoded_img = decoded_img.cpu().detach().numpy()
    decoded_img = np.transpose(decoded_img, (1,2,0))

    img_data = torch.squeeze(img_data)
    img_data = img_data.cpu().detach().numpy()
    img_data = np.transpose(img_data, (1,2,0))
    ax1.imshow(decoded_img)
    ax2.imshow(img_data)
    plt.savefig(str(batch_idx) + '.png')