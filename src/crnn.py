
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import string
import torch.nn.functional as F

# Define the character set (adjust based on your needs)
CHAR_SET = string.ascii_letters + string.digits + " -:"  # e.g., for "Adrion", "NISHIMIRWE", "4009113202", "50000 FRW"
CHAR2IDX = {char: idx for idx, char in enumerate(CHAR_SET)}
IDX2CHAR = {idx: char for idx, char in enumerate(CHAR_SET)}
BLANK_LABEL = len(CHAR_SET)  # For CTC loss

class CRNN(nn.Module):
    def __init__(self, img_height, num_chars, hidden_size=256):
        super(CRNN, self).__init__()
        
        # CNN layers for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (64, img_height/2, W/2)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (128, img_height/4, W/4)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # (256, img_height/8, W/4)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # (512, img_height/16, W/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # (512, img_height/32, W/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # (512, img_height/64, W/4)
        )
        
        # RNN layers for sequence modeling
        self.rnn = nn.Sequential(
            nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, batch_first=True),
        )
        
        # Fully connected layer for character prediction
        self.fc = nn.Linear(hidden_size * 2, num_chars + 1)  # +1 for CTC blank

    def forward(self, x):
        # x shape: (batch, channels=1, height=512, width=4096)
        batch_size = x.size(0)
        
        # CNN feature extraction
        conv = self.cnn(x)  # (batch, 512, height/64, W/4)
        # For height=512, height/64 = 8
        # For width=4096, width/4 = 1024
        # Shape: (batch, 512, 8, 1024)
        
        # Further downsample height to 1
        conv = F.max_pool2d(conv, (conv.size(2), 1))  # (batch, 512, 1, W/4)
        
        conv = conv.squeeze(2)  # (batch, 512, W/4)
        conv = conv.permute(0, 2, 1)  # (batch, W/4, 512)
        
        # RNN sequence modeling
        rnn_out, _ = self.rnn(conv)  # (batch, W/4, hidden_size*2)
        
        # Character prediction
        out = self.fc(rnn_out)  # (batch, W/4, num_chars+1)
        return out

class HTRDataset(Dataset):
    def __init__(self, image_dir, labels_file=None, target_height=512, target_width=4096):
        self.image_dir = image_dir
        self.target_height = target_height
        self.target_width = target_width
        
        # Load labels if provided (format: image_name, text)
        self.labels = {}
        if labels_file:
            with open(labels_file, 'r') as f:
                for line in f:
                    img_name, text = line.strip().split(',')
                    self.labels[img_name] = text
            self.image_names = list(self.labels.keys())
        else:
            self.image_names = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load preprocessed image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Ensure image matches target size
        img = img.astype(np.float32) / 255.0
        img = torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)
        
        if img_name in self.labels:
            label = self.labels[img_name]
            return img, label
        else:
            return img, None

def collate_fn(batch):
    """
    Custom collate function to handle batching of images and labels for CTC loss.
    """
    images = []
    labels = []
    label_lengths = []
    
    for img, label in batch:
        images.append(img)
        if label is not None:
            label_encoded = [CHAR2IDX[c] for c in label if c in CHAR2IDX]
            labels.extend(label_encoded)
            label_lengths.append(len(label_encoded))
    
    images = torch.stack(images, dim=0)  # (batch_size, 1, H, W)
    labels = torch.LongTensor(labels)  # (sum(label_lengths),)
    label_lengths = torch.LongTensor(label_lengths)  # (batch_size,)
    
    return images, labels, label_lengths

def ctc_decode(preds, blank_label=BLANK_LABEL):
    """
    Decode CTC output to text.
    """
    # preds: (batch, T, num_chars+1)
    preds = torch.softmax(preds, dim=2)
    preds = torch.argmax(preds, dim=2)  # (batch, T)
    
    # Debug: Print the predicted indices
    print(f"Predicted indices: {preds}")
    
    batch_size = preds.size(0)
    texts = []
    for b in range(batch_size):
        seq = []
        for t in range(preds.size(1)):
            char_idx = preds[b, t].item()
            if char_idx != blank_label and (not seq or seq[-1] != char_idx):
                seq.append(char_idx)
        text = ''.join(IDX2CHAR[idx] for idx in seq if idx in IDX2CHAR)
        texts.append(text)
    return texts

def train_crnn(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, labels, label_lengths) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)  # (batch, T, num_chars+1)
            outputs_for_loss = outputs.permute(1, 0, 2)  # (T, batch, num_chars+1) for CTC loss
            outputs_for_loss = outputs_for_loss.log_softmax(2)  # For CTC loss
            
            input_lengths = torch.full((images.size(0),), outputs_for_loss.size(0), dtype=torch.long).to(device)
            
            # Debug shapes
            print(f"Batch {batch_idx+1}:")
            print(f"images shape: {images.shape}")
            print(f"outputs shape: {outputs_for_loss.shape}")
            print(f"labels shape: {labels.shape}")
            print(f"input_lengths: {input_lengths}, shape: {input_lengths.shape}")
            print(f"label_lengths: {label_lengths}, shape: {label_lengths.shape}")
            
            loss = criterion(outputs_for_loss, labels, input_lengths, label_lengths)
            
            loss.backward()
            optimizer.step()
            
            # Debug: Decode predictions during training
            with torch.no_grad():
                predicted_texts = ctc_decode(outputs)
                print(f"Epoch [{epoch+1}/{num_epochs}], Predicted Text: {predicted_texts[0]}")
            
            total_loss += loss.item()
            if batch_idx % 1 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "../models/crnn_model.pth")
    print("Model saved to ../models/crnn_model.pth")

def infer_crnn(model, image_path, device, target_height=512, target_width=4096):
    model.eval()
    with torch.no_grad():
        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        img = img.astype(np.float32) / 255.0
        img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        img = img.to(device)
        
        # Run inference
        outputs = model(img)  # (1, T, num_chars+1)
        texts = ctc_decode(outputs)
        return texts[0]

if __name__ == "__main__":
    # Hyperparameters
    IMG_HEIGHT = 512
    IMG_WIDTH = 4096
    HIDDEN_SIZE = 256
    NUM_EPOCHS = 50  # Increased to give the model more time to learn
    BATCH_SIZE = 1  # Still 1 since we have only 1 image
    LEARNING_RATE = 0.0001  # Reduced for more gradual learning
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, criterion, optimizer
    model = CRNN(img_height=IMG_HEIGHT, num_chars=len(CHAR_SET), hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.CTCLoss(blank=BLANK_LABEL, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Dataset and DataLoader
    dataset = HTRDataset(
        image_dir="../data/processed",
        labels_file="../data/labels/labels.txt",
        target_height=IMG_HEIGHT,
        target_width=IMG_WIDTH
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Train the model
    train_crnn(model, dataloader, criterion, optimizer, DEVICE, NUM_EPOCHS)

    # Inference example
    test_image_path = "../data/processed/processed_demo.png"
    recognized_text = infer_crnn(model, test_image_path, DEVICE)
    print(f"Recognized Text: {recognized_text}")
