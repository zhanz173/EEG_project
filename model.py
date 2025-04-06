from backbone import EEGTransformerEncoder
import torch
import torch.nn as nn
import copy

class SimSiamProjector(nn.Modules):
    def __init__(self, dim, projection_size, hidden_size=4096, norm=nn.BatchNorm1d):
        super(SimSiamProjector, self).__init__()
        self.projector = nn.Sequential(
                            nn.Linear(dim, hidden_size, bias=False),
                            norm(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, hidden_size, bias=False),
                            norm(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, projection_size, bias=False),
                            norm(projection_size, affine=False)
                        )

    def forward(self, x):
        return self.projector(x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, norm=nn.BatchNorm1d):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.norm = norm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# time-freqency dual encoder
class TFC(nn.Module):
    def __init__(self, embeding_size=32, conv_kernel_size = [3,3,3,3],num_layers=4, num_heads=8, hidden_size=64, dropout=0.1):
        super(TFC, self).__init__()
        self.encoder_t = EEGTransformerEncoder(in_channels=1, kernel_size=conv_kernel_size, embeding_size=embeding_size, 
                                               stride=2, padding=1, num_layers=num_layers, num_heads=num_heads, 
                                               hidden_size=hidden_size, dropout=dropout)
        self.encoder_f = EEGTransformerEncoder(in_channels=1, kernel_size=conv_kernel_size, embeding_size=embeding_size, 
                                               stride=2, padding=1, num_layers=num_layers, num_heads=num_heads, 
                                               hidden_size=hidden_size, dropout=dropout)
        
        self.projector_t = MLP(embeding_size, hidden_size, embeding_size)

        self.projector_f = MLP(embeding_size, hidden_size, embeding_size)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        
    def forward(self, x_t, x_f):
        # freqency encoder
        h_f = self.encoder_f(x_f)
        h_f = torch.mean(h_f, dim=1)
        y_f = self.projector_f(h_f)

        # time encoder
        h_t = self.encoder_t(x_t)
        h_t = torch.mean(h_t, dim=1)
        y_t = self.projector_t(h_t)

        return h_t, y_t, h_f, y_f
    

#BYOL
# exponential moving average
def set_requires_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad = requires_grad

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class BYOL(nn.Module):
    def __init__(self, model, ema_decay=0.99):
        super(BYOL, self).__init__()
        self.online_encoder = model
        self.target_encoder = model
        self.target_encoder.eval()
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.ema_updater = EMA(ema_decay)

    def update_moving_average(self):
        update_moving_average(self.ema_updater, self.target_encoder, self.online_encoder)


# use BYOL to train the model
class EEG_BYOL(nn.Module):
    def __init__(self, base_model, embeding_size=32,hidden_size=64, projection_size=32,moving_average_decay=0.99):
        
        super(EEG_BYOL, self).__init__()
        self.net = base_model
        self.target_ema_updater = EMA(moving_average_decay)
        self.projector = MLP(embeding_size, hidden_size, projection_size)
        self.predictor = MLP(projection_size, hidden_size, projection_size)
        self.target_encoder = self._get_target_encoder()
        
    @torch.no_grad()
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.net)
        set_requires_grad(target_encoder, False)
        return target_encoder
    
    def __loss_function__(self, online_pred, target_pred):
        online_pred = nn.functional.normalize(online_pred, dim=-1)
        target_pred = nn.functional.normalize(target_pred, dim=-1)
        loss = -torch.mean(torch.sum(online_pred * target_pred, dim=-1))
        return loss
    
    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
    
    def forward(self, x1, x2, return_embedding = False):
        '''
        x1: augmentation 1
        x2: augmentation 2
        '''
        #online model
        online_y1 = self.net(x1)
        online_z1 = self.projector(online_y1)
        online_pred1 = self.predictor(online_z1)

        online_y2 = self.net(x2)
        online_z2 = self.projector(online_y2)
        online_pred2 = self.predictor(online_z2)

        if return_embedding:
            return online_y1, online_y2

        #target model
        with torch.no_grad():
            target_y1 = self.target_encoder(x1)
            target_z1 = self.projector(target_y1)
            target_pred1 = self.predictor(target_z1).detach()

            target_y2 = self.target_encoder(x2)
            target_z2 = self.projector(target_y2)
            target_pred2 = self.predictor(target_z2).detach()

        #loss
        loss1 = self.__loss_function__(online_pred1, target_pred2)
        loss2 = self.__loss_function__(online_pred2, target_pred1)
        loss = (loss1 + loss2) / 2
        return loss
