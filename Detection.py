import torch
import torch.nn as nn
from Model import FaceRecognizer
from tqdm import tqdm
import cv2
import os
import glob



class Detector:
    
    def __init__(self, image_files_path:str):
        
        self.facerecognizer, self.device = self.load_model()
        self.facerecognizer.model.classifier[6] = nn.Sequential()
        self.embeddings_path = self.create_embeddings_from_directory(image_files_path)
        
        self.embeddings = torch.load(self.embeddings_path)
        
        
    def detect(self, X:torch.Tensor):
        X = X.to(self.device)
        embs = self.facerecognizer(X)
        
        scores = torch.cosine_similarity(embs, self.embeddings, -1)

        return scores

    def load_model(self):
        """Load the trained model from lightning_logs/version_0"""
        # Find the checkpoint file
        checkpoint_path = glob.glob("lightning_logs_1/version_0/checkpoints/*.ckpt")
        
        if not checkpoint_path:
            raise FileNotFoundError("No checkpoint found in lightning_logs/version_0/checkpoints/")
        
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

    def create_embeddings_from_directory(self, directory_path, output_name=None, num_files=None):
        # Get list of audio files
        image_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        print("ImageFIles:", image_files)
        # Limit number of files if specified
        if num_files is not None:
            image_files = image_files[:num_files]
        
        # Create embeddings
        embeddings = []
        for image_path in tqdm(image_files, desc=f"Processing {len(image_files)} files"):
            # try:
            # Load audio file
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))
            img_pt = torch.tensor([img], dtype=torch.float32).transpose(1, -1)
            # Generate embedding
            with torch.no_grad():
                embedding = self.facerecognizer(img_pt.to(self.device))
            
            embeddings.append(embedding)
                
            # except Exception as e:
            #     print(f"Error processing {image_path}: {e}")
        
        # Stack embeddings and save
        if embeddings:
            embeddings_tensor = torch.stack(embeddings)
            print(embeddings_tensor.shape)
            
            # Determine output filename
            if output_name is None:
                folder_name = os.path.basename(os.path.normpath(directory_path))
                output_path = f"{folder_name}_embeddings.pt"
            else:
                output_path = f"{output_name}_embeddings.pt"
            
            # Save tensor
            torch.save(embeddings_tensor, output_path)
            print(f"Saved embeddings with shape {embeddings_tensor.shape} to {output_path}")
            return output_path
        else:
            print("No embeddings were created. Check if the directory contains valid audio files.")
            return None
        
if __name__ == "__main__":
    
    detector = Detector("./Test_Images")
    img = cv2.imread("./Test_Images/Virat Kohli_42.jpg")
    img_pt = torch.tensor([img], dtype=torch.float32).transpose(1, -1)
    out = detector.detect(img_pt)
    
    print(out)