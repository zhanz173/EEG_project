import torch
from model import BYOLModel

def BYOL_trainer_one_epoch(model:BYOLModel, dataloader, optimizer, ema_decay=0.99, device='cuda'):
    """
    BYOL trainer function.
    Args:
        model: The BYOL model.
        dataloader: The data loader for training data.
        optimizer: The optimizer for the model.
        ema_decay: Exponential moving average decay rate.
        device: Device to run the model on ('cuda' or 'cpu').
    """
    # Set the model to training mode
    total_loss = 0.0
    model.train()
    model.to(device)
    for batch_idx, (x_aug_1, x_aug_2) in enumerate(dataloader):
        # Move data to the specified device
        x_aug_1, x_aug_2 = x_aug_1.to(device), x_aug_2.to(device)
        optimizer.zero_grad()

        # Forward pass through the online encoder
        loss = model(x_aug_1, x_aug_2)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Update the moving average of the target encoder
        model.update_moving_average()

        total_loss += loss.item()

    return total_loss / len(dataloader)