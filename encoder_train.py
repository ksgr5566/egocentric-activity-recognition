import os
import random
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer
from torchvision.io import read_image


class FramePairsDataset(Dataset):
    def __init__(self, root_dir, split_txt_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_folders = []
        with open(split_txt_file, 'r') as f:
            for line in f:
                line = line.split()
                self.video_folders.append(line[0])
        
    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_path = os.path.join(self.root_dir, video_folder)
        frames = os.listdir(video_path)

        frame_pair = random.sample(frames, 2)  # Select 2 random frames
        frame1 = read_image(os.path.join(video_path, frame_pair[0]))
        frame2 = read_image(os.path.join(video_path, frame_pair[1]))

        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
        return frame1, frame2
    

class SimCLR(LightningModule):
    def __init__(self, base_encoder, temperature=0.5):
        super().__init__()
        self.base_encoder = base_encoder
        self.temperature = temperature
        self.projection_head = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 512))

    def forward(self, x):
        x = self.base_encoder(x)  # Base encoding
        x = self.projection_head(x)  # Projection head
        return F.normalize(x, dim=1)

    def contrastive_loss(self, x, y):
        # Contrastive loss as per SimCLR
        batch_size = x.shape[0]
        z = torch.cat([x, y], dim=0)
    
        # Change to calculate similarity matrix correctly
        z_norm = F.normalize(z, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.t()) / self.temperature
        sim_matrix = torch.exp(sim_matrix)

        # Exclude the diagonal elements in the sum by creating a new tensor
        sim_matrix_no_diag = sim_matrix - torch.diag(sim_matrix.diag())
        denominator = sim_matrix_no_diag.sum(dim=1)

        # Compute the mask for positive samples
        mask = torch.zeros((2 * batch_size, 2 * batch_size), device=self.device)
        mask[:batch_size, batch_size:] = torch.eye(batch_size, device=self.device)
        mask[batch_size:, :batch_size] = torch.eye(batch_size, device=self.device)

        # Compute the nominator
        nominator = (mask * sim_matrix_no_diag).sum(dim=1)

        loss = -torch.log(nominator / denominator).mean()
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        z_i = self.forward(x)
        z_j = self.forward(y)
        loss = self.contrastive_loss(z_i, z_j)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z_i = self.forward(x)
        z_j = self.forward(y)
        loss = self.contrastive_loss(z_i, z_j)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
        return optimizer
    
class PrintCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics['train_loss']
        print(f"Epoch {trainer.current_epoch}: Train Loss: {train_loss:.4f}")

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics['val_loss']
        print(f"Val Loss: {val_loss:.4f}")

def main():
    batch_size = 64
    
    transform = transforms.Compose([
        transforms.ToPILImage(),      
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FramePairsDataset("/content/rgb_dataset", "/content/train_split1.txt", transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

    test_dataset = FramePairsDataset("/content/rgb_dataset", "/content/test_split1.txt", transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

    base_encoder = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    base_encoder.fc = nn.Identity()

    for param in base_encoder.parameters():
        param.requires_grad = True

    model = SimCLR(base_encoder)

    trainer = Trainer(max_epochs=100, check_val_every_n_epoch=1, callbacks=[PrintCallback()])
    trainer.fit(model, dataloader, val_dataloaders=[test_dataloader])

if __name__ == "__main__":
    main()