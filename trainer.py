import torch
from models import EEG_BYOL, TFC

class BYOL_trainer:
    def __init__(self, model:EEG_BYOL, dataloader, optimizer, ema_decay=0.99, device='cuda'):
        """
        BYOL trainer class.
        Args:
            model: The BYOL model.
            dataloader: The data loader for training data.
            optimizer: The optimizer for the model.
            ema_decay: Exponential moving average decay rate.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.ema_decay = ema_decay
        self.device = device
        self.model.train()

    def train_one_epoch(self):
        """
        Train the model for one epoch.
        Returns:
            Average loss for the epoch.
        """
        total_loss = 0.0
        self.model.train()
        self.model.to(self.device)
        for batch_idx, batch in enumerate(self.dataloader):
            # Move data to the specified device
            x_aug_1, x_aug_2 = batch['EEG_Raw_Aug'].to(self.device), batch['EEG_Raw_Aug2'].to(self.device)
            self.optimizer.zero_grad()

            # Forward pass through the online encoder
            loss = self.model(x_aug_1, x_aug_2)

            # Backward pass and optimization step
            loss.backward()
            self.optimizer.step()

            # Update the moving average of the target encoder
            self.model.update_moving_average()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)


def TFC_trainer_on_epoch(model:TFC, dataloader, batch_size, optimizer, criteria, device):
    model.train()
    model.to(device)
    total_loss = 0
    for i, batch in enumerate(dataloader):
        if batch['EEG_Raw'].shape[0] < batch_size:
            continue
        data_t = batch['EEG_Raw'].to(device)
        data_f = batch['Freq'].to(device)
        data_t_aug = batch['EEG_Raw_Aug'].to(device)
        data_f_aug = batch['Freq_Aug'].to(device)

        optimizer.zero_grad()
        h_t, z_t, h_f, z_f = model(data_t, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(data_t_aug, data_f_aug)

        loss_t = criteria(h_t, h_t_aug)
        loss_f = criteria(h_f, h_f_aug)
        l_TF = criteria(z_t, z_f) # this is the initial version of TF loss
        loss = 0.2*(loss_t + loss_f) + l_TF
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(dataloader)