import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from PIL import Image
import torchvision.transforms as transforms
import os
import glob


class FaceRecognizer(pl.LightningModule):
    
    def __init__(self, output_size, lr=0.001):
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
        out = self(x)
        l = self.loss_fn(out, y)
        self.log('train_loss', l, prog_bar=True)
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
            "optimizer": optimizer
        }

def load_model():
    """Load the trained model from lightning_logs/version_0"""
    # Find the checkpoint file
    checkpoint_path = glob.glob("./checkpoints/*.ckpt")
    
    if not checkpoint_path:
        raise FileNotFoundError("No checkpoint found in /checkpoints/")
    
    # Use the latest checkpoint if multiple exist
    checkpoint_path = checkpoint_path[0]
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load the model with the same architecture and number of classes
    model = FaceRecognizer.load_from_checkpoint(checkpoint_path, output_size=2562)
    model.eval()  # Set to evaluation mode
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return model, device

def preprocess_image(image_path, device):
    """Preprocess a single image for inference"""
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor.to(device)

def test_single_image(model, image_path, device):
    """Run inference on a single image"""
    # Preprocess the image
    image_tensor = preprocess_image(image_path, device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get the predicted class
    _, predicted_class = torch.max(output, 1)
    confidence = torch.softmax(output, dim=1)[0, predicted_class.item()].item()
    
    return predicted_class.item(), confidence

def test_batch_from_features(model, device):
    """Run inference on a batch of features from the features.pt file"""
    # Load features
    features = torch.load("features.pt", weights_only=True)
    features = features.transpose(1, -1)  # Apply the same transformation as during training
    labels = torch.load("Labels.pt", weights_only=True)
    
    # Select a subset for testing
    test_size = 10  # Test 10 samples
    test_features = features[:test_size].to(device)
    test_labels = labels[:test_size].to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(test_features)
        _, predicted = torch.max(outputs, 1)
    
    # Calculate accuracy
    correct = (predicted == test_labels).sum().item()
    accuracy = correct / test_size
    
    print(f"Test accuracy on {test_size} samples: {accuracy * 100:.2f}%")
    print("Predictions:", predicted.cpu().numpy())
    print("Actual labels:", test_labels.cpu().numpy())
    
    return accuracy

def main():
    # Load the trained model
    model, device = load_model()
    print(f"Model loaded successfully and moved to {device}")
    
    # Test on batch of features
    accuracy = test_batch_from_features(model, device)
    
    # If you have specific image files to test, uncomment and use this
    """
    test_image_path = "path/to/your/test/image.jpg"
    if os.path.exists(test_image_path):
        predicted_class, confidence = test_single_image(model, test_image_path, device)
        print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
    """

if __name__ == "__main__":
    # main()
    model, device = load_model()
    # print(model.named_modules)
    
    model.model.classifier[6] = nn.Sequential()
    print(model.named_modules)
    # exit(0)
    import cv2
    img = cv2.imread("./Test_Images/Vijay Deverakonda_110.jpg")
    img2 = cv2.imread("./Test_Images/Virat Kohli_42.jpg")
    img3 = cv2.imread("./Test_Images/Zac Efron_90.jpg")
    
    img4 = cv2.imread("./Test_Images/MarkZuck.jpeg")
    
    img_pt = torch.tensor(img, dtype=torch.float32).transpose(0, -1)
    img_pt2 = torch.tensor(img2, dtype=torch.float32).transpose(0, -1)
    img_pt3 = torch.tensor(img3, dtype=torch.float32).transpose(0, -1)
    img_pt4 = torch.tensor(img4, dtype=torch.float32).transpose(0, -1)
    
    embs = model(img_pt.unsqueeze(0).to(device))
    embs2 = model(img_pt2.unsqueeze(0).to(device))
    embs3 = model(img_pt3.unsqueeze(0).to(device))
    embs4 = model(img_pt4.unsqueeze(0).to(device))
    print("similarity")
    print(torch.cosine_similarity(embs, embs, -1))
    print(torch.cosine_similarity(embs2, embs2, -1))
    print(torch.cosine_similarity(embs3, embs3, -1))
    print("disimilarity")
    print(torch.cosine_similarity(embs, embs4, -1))
    print(torch.cosine_similarity(embs2, embs4, -1))
    print(torch.cosine_similarity(embs3, embs4, -1))
    print(torch.cosine_similarity(embs, embs2, -1))
    print(torch.cosine_similarity(embs2, embs3, -1))
    print(torch.cosine_similarity(embs3, embs, -1))