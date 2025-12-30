# CNN-based encoder-decoder class to predict Land Surface Temperature (LST) from semantic features

###### import libraries ######
# Data Science/ML libraries
import torch.nn as nn


class LSTPredictor(nn.Module):
    """
    Predictor network to estimate Land Surface Temperature from semantic features.
    
    Architecture: CNN-based encoder-decoder for pixel-wise LST prediction.
    """
    
    def __init__(self, in_channels, hidden_dims=[64, 128, 256], out_channels=1):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        
        # Encoder
        encoder_layers = []
        prev_dim = in_channels
        
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(prev_dim, h_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, h_dim),
                nn.SiLU(),
                nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, h_dim),
                nn.SiLU()
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        
        for i in reversed(range(len(hidden_dims))):
            h_dim = hidden_dims[i]
            next_dim = hidden_dims[i-1] if i > 0 else hidden_dims[0]
            
            decoder_layers.extend([
                nn.ConvTranspose2d(h_dim, next_dim, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, next_dim),
                nn.SiLU(),
                nn.Conv2d(next_dim, next_dim, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, next_dim),
                nn.SiLU()
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Final prediction head
        self.head = nn.Conv2d(hidden_dims[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input semantic tensor [B, in_channels, H, W]
            
        Returns:
            LST prediction [B, 1, H, W]
        """
        # Encode
        z = self.encoder(x)
        
        # Decode
        out = self.decoder(z)
        
        # Predict LST
        lst_pred = self.head(out)
        
        return lst_pred