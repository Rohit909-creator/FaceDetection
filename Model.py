import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torchvision

val_losses = []
train_losses = []
# model = torchvision.models.vgg11()
# print(model.named_modules)
# torch.set_default_device('cuda')
class FaceRecognizer(pl.LightningModule):
    
    def __init__(self, output_size, lr=0.01):
        super().__init__()
        self.lr = lr
        self.model = torchvision.models.vgg11()
        self.model.classifier[6] = nn.Linear(4096, output_size)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, X):
        out = self.model(X)
        return out
    
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        # x = x.cuda()
        # y = y.cuda()
        out = self(x)
        l = self.loss_fn(out, y)
        train_losses.append(l.item())
        self.log('train loss', l, prog_bar=True)
        return l
    
    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        out = self(x)
        l = self.loss_fn(out, y)
        val_losses.append(l.item())
        self.log("val_loss", l, prog_bar=True)        
        return l
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": None,
            # "monitor": "val_loss"
        }
        
        
        
class MemoryBased(pl.LightningModule):
    
    def __init__(self, width, hieght, output_size, lr=0.01):
        super().__init__()
        self.lr = lr
        self.cnn = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2, padding=0)
        self.relu = nn.ReLU()
        self.persistent_memory = PersistantMemory(width, hieght, 3)
        self.restofthemodel = torchvision.models.vgg11()
        self.restofthemodel.classifier[6] = nn.Linear(4096, output_size)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, X):
        out = self.cnn(X)
        out = self.maxpool(out)
        out = self.relu(out)
        memory = self.persistent_memory()
        out = out - memory # This will be plotted and observed
        # print('Shape of out:', out.shape)
        out = self.restofthemodel(out)
        return out
    
    def training_step(self, batch, batch_idx):
            
        x, y = batch
        # x = x.cuda()
        # y = y.cuda()
        out = self(x)
        l = self.loss_fn(out, y)
        
        self.log('train loss', l, prog_bar=True)
        return l
    
    def validation_step(self, batch, batch_idx):
            
        x, y = batch
        out = self(x)
        l = self.loss_fn(out, y)
        self.log("val_loss", l, prog_bar=True)        
        return l
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": None,
            # "monitor": "val_loss"
        }
        
        
class PersistantMemory(pl.LightningModule):
    
    def __init__(self, width, hieght, channels=3, lr=0.01):
        super().__init__()
        self.x = nn.Parameter(torch.randn((1, channels, width, hieght)))
        self.lr = lr
        self.cnn = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2, padding=0)
        self.relu = nn.ReLU()
        
    def forward(self):
        
        out = self.cnn(self.x)
        out = self.maxpool(out)
        out = self.relu(out)
        return out        
# Instead of setting default device globally, be explicit with device placement
# Remove or comment out: torch.set_default_device('cuda')

def create_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data and move to device
    X1 = torch.load("features.pt", weights_only=True)  # Add weights_only=True to avoid warning
    X1 = X1.transpose(1, -1)
    print('Shape of data:', X1.shape)
    Y = torch.load("Labels.pt", weights_only=True)  # Add weights_only=True to avoid warning
    print(Y.shape)
    
    # Keep data on CPU for now (will move to GPU during training)
    return TensorDataset(X1, Y)

def get_dataloaders(batch_size=64):
    dataset = create_data()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Create a generator that matches your device
    generator = torch.Generator()
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# # Initialize and train model
def train_model():
    # input_size, hidden_size, output_size = 256, 512, 2
    # model = FaceRecognizer(2562)
    model = MemoryBased(160, 160 ,output_size=2562)
    train_loader, val_loader = get_dataloaders()

    trainer = pl.Trainer(max_epochs=500,
                        enable_progress_bar=True,  # Disable default tqdm ba
                        num_nodes=1,
                        enable_checkpointing=True
                        )
    
    trainer.fit(model, train_loader, val_loader)

# def train_model():
#     # Define model
#     model = FaceRecognizer(2562)
    
#     # Get dataloaders
#     train_loader, val_loader = get_dataloaders()
    
#     # Add callbacks for early stopping and model checkpointing
#     early_stop_callback = pl.callbacks.EarlyStopping(
#         monitor='val_loss',
#         patience=10,
#         verbose=True,
#         mode='min'
#     )
    
#     checkpoint_callback = pl.callbacks.ModelCheckpoint(
#         monitor='val_loss',
#         dirpath='checkpoints/',
#         filename='face-recognizer-{epoch:02d}-{val_loss:.2f}',
#         save_top_k=1,
#         mode='min',
#         save_last=True
#     )
    
#     # Add learning rate scheduler
#     lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    
#     trainer = pl.Trainer(
#         max_epochs=100,
#         enable_progress_bar=True,
#         callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
#         enable_checkpointing=True,
#         num_nodes=1,  # Keep if you're using multi-node training
#         gradient_clip_val=0.5,  # Prevent exploding gradients
#         log_every_n_steps=10,
#         # If using GPU, you might want to add:
#         accelerator='auto',  # Automatically detect available hardware
#         devices='auto'  # Use all available devices
#     )
    
#     trainer.fit(model, train_loader, val_loader)
    
#     # Return the path to the best checkpoint
#     best_model_path = checkpoint_callback.best_model_path
#     print(f"Best model saved at: {best_model_path}")
#     return best_model_path    
if __name__ == "__main__":
    
    # train_dataloader, _ = get_dataloaders()
    # print('Shape of data:', train_dataloader.dataset.tensors[0].shape)
    train_model()
    import json
    graph = {'train': train_losses, 'val':val_losses}
    with open('Graph.json', 'w') as f:
        s = json.dumps(graph)
        f.write(s)
    # model = MemoryBased(256, 256, 3)
    # x = torch.randn((1, 3, 256, 256))
    # y = model(x)
    # print(y.shape)