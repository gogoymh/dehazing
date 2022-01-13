import torch.nn as nn

class Hazenet(nn.Module):
    def __init__(self):
        super(Hazenet, self).__init__()
        
        self.decoder = nn.Sequential(
            
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(),
            
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(),
            
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU()
            
        )
        
        self.encoder = nn.Sequential(
            
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(),
            
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(),
            
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
            
        )
        

    def forward(self, Hazy):
        output = self.decoder(Hazy)
        output = self.encoder(output)
        output = output + Hazy
        return output