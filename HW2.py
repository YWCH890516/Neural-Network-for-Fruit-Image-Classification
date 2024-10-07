import os
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Set GPU or CPU to run this code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} to run this code')

# Set seed
def set_all_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_all_seed(999)

# Set PCA
use_pca = True
n_components = 2
if use_pca:
    pca = PCA(n_components)  # Dimensionality reduction

# Data augmentation transformation
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# Create Dataset and DataLoader
class FruitClassificationDataset(Dataset):
    def __init__(self, img_dir, name, use_pca=False, transform=None):
        assert name in ['train', 'test'], "name must be 'train' or 'test'"
        self.name = name
        name_map = {
            'train': 'Data_train',
            'test': 'Data_test'
        }
        self.img_dir = os.path.join(img_dir, name_map.get(self.name))
        self.data_type = ['Carambula', 'Lychee', 'Pear']
        self.data_num_per_class = 490 if self.name == 'train' else 166
        self.use_pca = use_pca
        self.transform = transform
        self.images, self.labels = self.load_data()

        if self.use_pca:
            self.images_pca = self.get_PCA_features()

    def load_data(self):
        images = []
        labels = []
        for label, type_name in enumerate(self.data_type):
            for i in range(self.data_num_per_class):
                fname = os.path.join(self.img_dir, type_name, f'{type_name}_{self.name}_{i}.png')
                image = np.array(Image.open(fname), dtype=np.float32)[..., 0] / 255.
                images.append(image)
                labels.append(label)
        return np.array(images), labels

    def get_PCA_features(self):
        images_reshape = self.images.reshape(self.images.shape[0], -1)
        if self.name == 'train':
            return pca.fit_transform(images_reshape)
        else:
            return pca.transform(images_reshape)

    def __getitem__(self, index):
        images = self.images_pca if self.use_pca else self.images
        return torch.tensor(images[index], dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self):
        return len(self.images)

class TwoLayerNN(nn.Module):
    NUM_CLASSES = 3
    def __init__(self, input_dim, hiden_size1 = 512, output_dim=NUM_CLASSES):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hiden_size1)
        #self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hiden_size1, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)
        return x

class ThreeLayerNN(nn.Module):
    NUM_CLASSES = 3
    def __init__(self, input_dim, hidden_size1=512, hidden_size2=256, output_dim=NUM_CLASSES):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_dim)
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
"""
# Handle class imbalance by calculating weights
def calculate_class_weights(dataset):
    class_counts = np.bincount(dataset.labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[dataset.labels]
    return sample_weights
"""
    
def train_model(dataloader, model, loss_fn, optimizer, num_epochs, device, use_pca= False):
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        size = len(dataloader.dataset)
        epoch_loss = 0
        correct = 0

       

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            if use_pca:
                images = images.reshape(images.shape[0], -1)

            # Compute prediction error
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Backpropagation with L2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            """
             # Backpropagation
            model.zero_grad()  # Reset gradients
            loss.backward()  # Compute gradients
            """

            epoch_loss += loss.item() * images.size(0)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
     
        avg_epoch_loss = epoch_loss / size
        avg_accuracy = correct / size

        train_losses.append(avg_epoch_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return model , train_losses

def test_model(model, dataloader, loss_fn, device, use_pca = False ):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            if use_pca:
                images = images.reshape(images.shape[0], -1)

            outputs = model(images)

            loss =loss_fn(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 

            probs = F.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    accuracy = correct / total
    epoch_loss = running_loss / len(dataloader.dataset)

    print(f'Loss: {epoch_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

    return np.concatenate(all_preds), np.concatenate(all_labels), accuracy, epoch_loss
""""
def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs Epoch')

    plt.show()

def plot_decision_regions(X, y, model, title, device):
    # Define the mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Flatten the mesh grid to shape (num_samples, num_features)
    mesh_input = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    print(f"Mesh grid shape: {mesh_input.shape}")  # Debugging

    # Predict the labels of mesh grid
    with torch.no_grad():
        Z = model(mesh_input).argmax(dim=1).cpu().numpy()

    Z = Z.reshape(xx.shape)

    # Plot the decision boundary using seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(Z, xticklabels=False, yticklabels=False, cmap='Spectral', alpha=0.3, zorder=1,
                cbar=False, linewidth=0, rasterized=True)
    
    # Overlay the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', s=25, edgecolor='k', cmap=plt.cm.Spectral, zorder=2)

    # Add title and legend
    plt.title(title)
    plt.legend(handles=scatter.legend_elements()[0], labels=['Carambula', 'Lychee', 'Pear'], loc='upper right')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()
"""
def check_model_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if use_pca:
                images = images.reshape(images.shape[0], -1)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    print("Unique predictions in test data:", np.unique(all_preds))
    print("Unique labels in test data:", np.unique(all_labels))

   

def plot_decision_regions(X, y, model, title, device):
    # Define the mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Flatten the mesh grid to shape (num_samples, num_features)
    mesh_input = np.c_[xx.ravel(), yy.ravel()]

    
    # Process the mesh grid in batches
    batch_size = 10000
    mesh_output = []
    with torch.no_grad():
        for i in range(0, len(mesh_input), batch_size):
            batch = torch.tensor(mesh_input[i:i + batch_size], dtype=torch.float32).to(device)
            batch_output = model(batch).argmax(dim=1).cpu().numpy()
            mesh_output.append(batch_output)
    
    # Concatenate all batch outputs
    Z = np.concatenate(mesh_output)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary using seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(Z, xticklabels=False, yticklabels=False, cmap='Spectral', alpha=0.3, zorder=1,
                cbar=False, linewidth=0, rasterized=True)
    
    # Overlay the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', s=25, edgecolor='k', cmap=plt.cm.Spectral, zorder=2)

    # Add title and legend
    plt.title(title)
    plt.legend(handles=scatter.legend_elements()[0], labels=['Carambula', 'Lychee', 'Pear'], loc='upper right')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

if __name__ == '__main__':
 
    # Create datasets
    df_train = FruitClassificationDataset('D:\\清大\\Machine Learning\\HW2', 'train', use_pca=True,transform=data_transforms)
    df_test = FruitClassificationDataset('D:\\清大\\Machine Learning\\HW2', 'test', use_pca=True)
    """
    # Handle class imbalance with WeightedRandomSampler
    sample_weights = calculate_class_weights(df_train)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    """
    # Create DataLoader
    batch_size = 32
    df_train_loader = DataLoader(df_train, batch_size=batch_size,shuffle=True , num_workers=4)
    df_test_loader = DataLoader(df_test, batch_size=batch_size, shuffle=False, num_workers=4)

    # Set model
    IMAGE_H = 32
    IMAGE_W = 32
    input_dim = n_components if use_pca else IMAGE_H * IMAGE_W
    loss_fn = nn.CrossEntropyLoss()
   
    # Define and train two-layer model
    model = TwoLayerNN(input_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trained_model,train_losses= train_model(df_train_loader, model, loss_fn, optimizer, num_epochs=20, device = device, use_pca= True)
    test_preds, test_labels, test_acc, test_loss = test_model(trained_model, df_test_loader, loss_fn, device, use_pca = False )
    #print(f'Two-Layer Model Accuracy: {accuracy * 100:.2f}%')
     # Plot training loss for two-layer model
    plt.figure()
    plt.plot(range(1, 21), train_losses, label='Two-Layer NN')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve for Two-Layer NN')
    plt.legend()
    plt.show()
    
    print('-' * 150)
    print('these are 3 layer net')
    # Define and train three-layer model
    model2 = ThreeLayerNN(input_dim).to(device)
    optimizer = torch.optim.SGD(model2.parameters(), lr=1e-3)  # Ensure optimizer is updated
    trained_model2 , train_losses2 = train_model(df_train_loader, model2, loss_fn, optimizer, num_epochs=20, device = device, use_pca= True)
    test_preds2, test_labels2, test_acc2, test_loss2 = test_model(trained_model2, df_test_loader, loss_fn, device, use_pca = False)
    #print(f'Three-Layer Model Accuracy: {accuracy2 * 100:.2f}%')
    # Plot training loss for three-layer model
    plt.figure()
    plt.plot(range(1, 21), train_losses2, label='Three-Layer NN')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve for Three-Layer NN')
    plt.legend()
    plt.show()
    
   
    """
    # Get training data for decision region plotting
    train_images, train_labels = next(iter(df_train_loader))
    if use_pca:
    
    train_images, train_labels = train_images.to(device), train_labels.to(device)
    print(f"Train image shape: {train_images.shape}")
""" 

    trained_model.eval()
    # Plot decision regions
    plot_decision_regions(df_train.images_pca, np.array(df_train.labels), trained_model, "Two Layer Network - Training Data",device)
    plot_decision_regions(df_train.images_pca, np.array(df_train.labels), trained_model2, "Three Layer Network - Training Data",device)
    #plot_decision_regions(train_images.cpu().numpy(), train_labels.cpu().numpy(), trained_model2, "Three Layer Network - Training Data",device)
    """
    # Get testing data for decision region plotting
    test_images, test_labels = next(iter(df_test_loader))
    if use_pca:
     test_images = test_images.reshape(test_images.shape[0], -1)
    test_images, test_labels = test_images.to(device), test_labels.to(device)
    print(f"Test images shape: {test_images.shape}")  # Debugging
    """
    # Plot decision regions
    plot_decision_regions(df_test.images_pca, np.array(df_test.labels), trained_model, "Two Layer Network - Testing Data",device)
    plot_decision_regions(df_test.images_pca, np.array(df_test.labels), trained_model2, "Three Layer Network - Testing Data",device)
    #plot_decision_regions(test_images.cpu().numpy(), test_labels.cpu().numpy(), trained_model2, "Three Layer Network - Testing Data",device)

      

    check_model_predictions(trained_model, df_test_loader, device)
    check_model_predictions(trained_model2, df_test_loader, device)
    #check_model_predictions(trained_model2, device)
      # Optional: Display confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(test_labels, test_preds.argmax(axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df_train.data_type)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    cm2 = confusion_matrix(test_labels2, test_preds2.argmax(axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=df_train.data_type)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()