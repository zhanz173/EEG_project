from backbone import EEGTransformerEncoder
import torch
import torch.nn as nn

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
        
        self.projector_t = nn.Sequential(
            nn.Linear(embeding_size, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32)
        )

        self.projector_f = nn.Sequential(
            nn.Linear(embeding_size, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32)
        )

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