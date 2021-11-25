import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from VAE_helper import set_all_seeds
from VAE_helper import VAE
import torch.nn.functional as F
import time
####################
#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
####################
#parameters
Batch_size = 256
random_seed = 1
learning_rate = 2e-4
training_epochs = 100
set_all_seeds(random_seed)
reconstruction_weight = 1
####################
# prepare data
transform = transforms.Compose([transforms.CenterCrop(128),
                                transforms.ToTensor()])
dataset = datasets.ImageFolder('imgs', transform=transform)
train_DataLoader = DataLoader(dataset, batch_size=Batch_size, shuffle=True)
####################
# MODEL initiation
model = VAE()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
##########################
#training

start_time = time.time()
for epoch in range(training_epochs):
    for batch_idx, img_data in enumerate(train_DataLoader):
        img_data = img_data[0].to(device)
        encoded_latent, z_mean, z_log_var, decoded_img = model(img_data)

        kl_div = 1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var)
        kl_div = -0.5 * torch.sum(kl_div, axis=1)
        kl_div = kl_div.mean()  # average over batch dimension
        reconstruction_err = F.mse_loss(decoded_img, img_data, reduction='none')
        reconstruction_err = reconstruction_err.view(Batch_size, -1).sum(axis=1)  # sum over pixels
        reconstruction_err = reconstruction_err.mean()  # average over batch dimension

        loss = reconstruction_weight * reconstruction_err + kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                  % (epoch + 1, training_epochs, batch_idx,
                     len(train_DataLoader), loss))
            print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

        torch.save(model.state_dict(), 'saved_model/' + 'model_' + str(epoch) + '.pth')


#########################

