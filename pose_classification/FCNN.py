import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import argparse


# 設置 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 自定義 Dataset
class PoseDataset(Dataset):
    def __init__(self, data):
        self.data = []
        self.labels = []
        for label, samples in data.items():
            for sample in samples:
                self.data.append(sample)
                self.labels.append(int(label) - 1)  # Convert label to int

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 2. FCNN 模型
class FCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.layers(x)
        return x.squeeze(1)

# 3. 加載 JSON 數據
def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# 4. 訓練模型
def train_model(model, train_loader, test_loader, num_epochs, criterion, optimizer, scheduler, logger, path, patience=5):
    best_acc = float(0.0)
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy for training
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Training accuracy
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy for validation
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        # Validation accuracy
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        # Log losses and accuracies
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                    f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Early stopping check
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), f"{path}/best_model.pth")
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Best model saved with Val Acc {best_acc:.4f}")
            best_preds = all_preds
            best_labels = all_labels
            
        # Adjust learning rate
        scheduler.step(avg_val_loss)
        torch.save(model.state_dict(), f"{path}/last_epoch.pth")
        if epochs_no_improve >= patience:
            logger.info("Early stopping!")
            break

    return model, best_preds, best_labels, train_losses, val_losses, train_accuracies, val_accuracies

# 5. 繪製混淆矩陣
def plot_confusion_matrix(y_true, y_pred, classes, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(f"{path}/confusion_matrix.png")

# 6. 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data")
    parser.add_argument("--feature", type=int, required=True, help="nums of feature")
    parser.add_argument("--name", type=str,  required=True, help="file name in FCNN")
    parser.add_argument("--batch", type=int, default="64")
    parser.add_argument("--test_member", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default= 10)
    parser.add_argument("--side", type=str, required=True, help="file name in FCNN")
    
    args = parser.parse_args()
    
    batch = args.batch
    test_member = args.test_member
    num_classes = args.num_classes
    side = args.side
    feature = args.feature
    name = args.name
    
    # 創建結果資料夾
    if not os.path.exists("result/FCNN"):
        os.makedirs("result/FCNN")
    
    
    
    if not os.path.exists(f"result/FCNN/{name}"):
        os.makedirs(f"result/FCNN/{name}")
        
    path = f"result/FCNN/{name}/{test_member}"
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    
    # 設置Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
     
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
     
    file_handler = logging.FileHandler(os.path.join(path, "log.log"))
    file_handler.setLevel(logging.INFO)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 加載數據
    train_data = load_data(f"dataset_input/{test_member}_train.json")  # JSON格式數據: {1:[[23個長度],[23]...], 2...}
    test_data = load_data(f"dataset_input/{test_member}_test.json")
    

    train_dataset = PoseDataset(train_data)
    test_dataset = PoseDataset(test_data)


    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    
    
    # 模型、損失函數、優化器和學習率調整器

    
    model = FCNN(feature, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # 訓練模型
    model, y_pred, y_true, train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, test_loader, num_epochs=50,
                                                                  criterion=criterion, optimizer=optimizer, 
                                                                  scheduler=scheduler, logger=logger, path=path, patience=5)

    # 保存訓練和驗證損失圖
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig(f"{path}/loss_curve.png")
    
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.savefig(f"{path}/accuracy_curve.png")

    # 混淆矩陣
    plot_confusion_matrix(y_true, y_pred, [str(i) for i in range(1, num_classes+1)], path)
