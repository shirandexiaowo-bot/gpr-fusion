# %% [markdown]
## Import packages
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# %%
# PyTorch Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve, spectrogram
from tqdm import tqdm
from skimage.transform import resize
# %% [markdown]
# # Prepare Dataset

# %%
# Load data with shape: [number of traces, samples per trace, number of center frequencies]
# Define image directory path
folder_path = r'./data'
# Read input images
img_1 = plt.imread(os.path.join(folder_path, '1.png'))
img_2 = plt.imread(os.path.join(folder_path, '3.png'))
img_3 = plt.imread(os.path.join(folder_path, '2.png'))

# Convert RGB images to grayscale if needed
if len(img_1.shape) == 3:
    img_1 = np.mean(img_1, axis=2)
if len(img_2.shape) == 3:
    img_2 = np.mean(img_2, axis=2)
if len(img_3.shape) == 3:
    img_3 = np.mean(img_3, axis=2)

# Standardize the input images (zero mean, unit variance)
img_1 = (img_1 - np.mean(img_1)) / np.std(img_1)
img_2 = (img_2 - np.mean(img_2)) / np.std(img_2)
img_3 = (img_3 - np.mean(img_3)) / np.std(img_3)

# %% 
# Plot intensity distribution histograms of input images
plt.hist(img_1.flatten(), bins=50, alpha=0.5, label='img_1')
plt.hist(img_2.flatten(), bins=50, alpha=0.5, label='img_2')
plt.legend()
plt.show()
# %% 
# Stack multi-frequency images into a 3D array
img_stacked = np.stack((img_1, img_2), axis=2)

# Visualize stacked channels (optional)
plt.imshow(img_stacked[:, :, 0], cmap='gray')
plt.title('2G Hz')
plt.show()

plt.imshow(img_stacked[:, :, 1], cmap='gray')
plt.title('Cycle-GAN')
plt.show()
# %%
# Transpose data dimensions for GPR processing
data_full = img_stacked.transpose((1,0,2))
ntr, ns, nfreq = data_full.shape
# Define time window thresholds for different frequency components
thershold_high   = [0, 450]
thershold_low    = [0, ns]

# %%
def calculate_weight(data_full, ns_seg=100, overlap=0.5):
    """
    Calculate adaptive fusion weights for multi-frequency GPR data
    based on frequency features and spatial gradients.
    Args:
        data_full: Input GPR data with shape [ntr, ns, nfreq]
        ns_seg: Window length for sliding window calculation
        overlap: Overlap ratio between adjacent windows
    Returns:
        weight: Normalized fusion weights for each sample point
    """
    ntr, ns, nfreq = data_full.shape
    # Extract valid data within specified time windows
    data_used = np.zeros_like(data_full)
    data_used[:, thershold_high[0]:thershold_high[1], 0] = data_full[:, thershold_high[0]:thershold_high[1], 0]
    data_used[:, thershold_low[0]:thershold_low[1], 1] = data_full[:, thershold_low[0]:thershold_low[1], 1]
    
    # Calculate frequency-domain indicator using spectrogram
    def frequency_indicator(data):
        if data.ndim > 1:
            raise ValueError("Input data to `frequency_indicator` must be 1D.")
        freqs, times, Sxx = spectrogram(data, fs=1.0, nperseg=128, noverlap=64)
        indicator = np.sum(Sxx * freqs[:, np.newaxis]**2, axis=0)
        time_resampled = np.interp(np.linspace(0, 1, ns), np.linspace(0, 1, len(times)), indicator)
    
        # Normalize to avoid zero division
        time_resampled = time_resampled / (np.sum(time_resampled) + 1e-8)
    
        return time_resampled
    
    # Calculate spatial gradient using Laplacian kernel
    kernel_space = np.array([1, -4, 6, -4, 1]).reshape(-1, 1)
    spatial_grad = np.zeros_like(data_full)
    for ifreq in range(nfreq):
        for itr in range(ntr):
            spatial_grad[itr, :, ifreq] = np.abs(convolve(data_used[itr, :, ifreq], kernel_space[:, 0], mode='same'))
    
    # Compute combined indicator: frequency features + spatial gradient
    indicator = np.zeros_like(data_full)
    for ifreq in range(nfreq):
        for itr in range(ntr):
            freq_feat = frequency_indicator(data_used[itr, :, ifreq])
            spatial_feat = spatial_grad[itr, :, ifreq]
            indicator[itr, :, ifreq] = freq_feat * 0.5 + spatial_feat
    
    # Sliding window configuration
    ns_overlap = round(overlap * ns_seg)
    nseg = int(np.fix((ns - ns_overlap) / (ns_seg - ns_overlap)))
    i_start = np.arange(stop=nseg * (ns_seg - ns_overlap), step=ns_seg - ns_overlap)
    i_end = i_start + ns_seg
    if i_end[-1] != ns:
        i_start = np.r_[i_start, ns - ns_seg]
        i_end = np.r_[i_end, ns]
    nseg = len(i_start)
    
    # Compute averaged weights with sliding window
    weight = np.zeros_like(data_full)
    norm_factor = np.zeros_like(data_full)
    for iseg in range(nseg):
        weight[:, i_start[iseg]:i_end[iseg], :] += np.expand_dims(
            np.sum(indicator[:, i_start[iseg]:i_end[iseg], :], axis=1), axis=1
        ) / ns_seg
        norm_factor[:, i_start[iseg]:i_end[iseg], :] += 1
    weight = weight / (norm_factor + 1e-8)
    
    # Normalize weights across frequency dimension
    weight_sum = np.sum(weight, axis=2, keepdims=True)
    weight = weight / (weight_sum + 1e-8)
    
    return weight


# %%
# Use the first 100 traces for model training
data_train = data_full[:100, :, :]

# Calculate adaptive fusion weights for training data
weight_train = calculate_weight(data_train, ns_seg=100, overlap=0.9)

# %%
# Visualize calculated weights for a test trace
test_trace = 50
dt = 0.05*1e-9
fig=plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(np.arange(ns)*dt*1e9, weight_train[test_trace, :, 0], label='High')
plt.plot(np.arange(ns)*dt*1e9, weight_train[test_trace, :, 1], label='Cycle-GAN')
ax.legend()
plt.xlabel('Time [ns]')
plt.ylabel('Normalized weight')

# %% [markdown]
# # Neural Network Configurations and Training Parameters

# %%
# Transformer architecture hyperparameters
d_model = 32
nhead = 2
dim_feedforward = 64
num_layers = 2
max_signal_length = 4096

# Training configuration
config = {
    'n_epochs': 50,
    'batch_size': 16,
    'learning_rate': 1e-3,
    'save_path': './models/GPRfusionformer_base.ckpt',
}

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer
    Args:
        d_model: Dimension of model features
        max_seq_length: Maximum sequence length
    """
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:,:x.size(1)]

class GPRfusionformer(nn.Module):
    """
    Transformer-based GPR data fusion model
    Args:
        input_size: Number of input frequency channels
        d_model: Feature dimension
        nhead: Number of attention heads
        dim_feedforward: Feed-forward network dimension
        num_layers: Number of Transformer encoder layers
        max_signal_length: Maximum input signal length
    """
    def __init__(self, input_size, d_model, nhead, dim_feedforward, num_layers, max_signal_length):
        super(GPRfusionformer, self).__init__()
        
        self.linear_embedding = nn.Linear(input_size, d_model, bias=False)
        self.positional_encoding = PositionalEncoding(d_model, max_signal_length)
        
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, 
                                       nhead=nhead, 
                                       dim_feedforward=dim_feedforward, 
                                       batch_first=True, 
                                       norm_first=True), 
            num_layers=num_layers)
        
        self.linear = nn.Linear(d_model, input_size)
        
    def forward(self, x):
        emb = self.positional_encoding(self.linear_embedding(x))
        emb = self.encoder_layers(emb)
        weight = F.softmax(self.linear(emb), dim=-1)
        y = torch.sum(torch.mul(x, weight), dim=-1)
        return y

# %% [markdown]
# # Trainer Functions
def trainer(train_loader, model, config, device):
    """
    Training function for GPRfusionformer
    Args:
        train_loader: Training data loader
        model: GPRfusionformer model
        config: Training configuration dictionary
        device: Computing device
    Returns:
        loss_record: Training loss history
    """
    # Weighted MSE loss function
    def loss_fn(y_true, weight, y_pred):
        squared_difference = weight * torch.square((y_true - y_pred.unsqueeze(2)))
        loss_sum = torch.sum(torch.mean(squared_difference, (1, 0)))
        return loss_sum

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) 
    
    # Create model save directory
    if not os.path.isdir('./models'):
        os.mkdir('./models') 
    
    def closure():
        optimizer.zero_grad()
        x = data
        y = weight
        x = x.to(device)
        y = y.to(device)
        pred = model(x)             
        loss = loss_fn(x, y, pred)
        loss.backward()
        return loss
    
    
    loss_record = []
    for epoch in range(config['n_epochs']):
        model.train()

        train_pbar = tqdm(train_loader, position=0, leave=True)
        loss_epoch = []
        for data, weight in train_pbar:
            loss = optimizer.step(closure=closure)
            
            train_pbar.set_description(f'Epoch [{epoch+1}/{config["n_epochs"]}]')
            train_pbar.set_postfix({'loss': loss.item()})
            
            loss_epoch.append(loss.item())
            
        loss_record.append(np.mean(loss_epoch))
    
    torch.save(model.state_dict(), config['save_path'])
    print('Saving model with loss {:.3f}...'.format(loss_record[-1]))
    
    return loss_record

def predict(test_loader, model, device):
    """
    Inference function for trained model
    Args:
        test_loader: Test data loader
        model: Trained GPRfusionformer
        device: Computing device
    Returns:
        preds: Fusion results
    """
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds

# %% [markdown]
# # Training Execution

class GPRfusion_Dataset(Dataset):
    """
    Custom Dataset class for GPR fusion task
    Args:
        x: Multi-frequency GPR data
        y: Fusion weights (None for inference)
    """
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

# %%
# Create dataset and dataloader for training
train_dataset = GPRfusion_Dataset(data_train, weight_train)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

# Initialize model
model = GPRfusionformer(input_size=nfreq, 
                        d_model=d_model, 
                        nhead=nhead, 
                        dim_feedforward=dim_feedforward, 
                        num_layers=num_layers, 
                        max_signal_length=max_signal_length).to(device)

# %%
# Start training
loss_record = trainer(train_loader, model, config, device)

# %%
# Plot training loss curve
plt.figure(figsize=[10, 5])
plt.plot(loss_record)
plt.title('loss plot')
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.grid(True)
plt.show()

# %% [markdown]
# # Inference and Fusion Results
# Run inference on full dataset
test_dataset = GPRfusion_Dataset(data_full)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
y_pred = predict(test_loader, model, device)

# %%
# Visualize input and fusion results
v_min, v_max = -4, 3 
c_map = 'gray'
fig = plt.figure(figsize=[25, 60])
dx = 0.008
dt = 0.04*1e-9

ax1 = plt.subplot(411)
plt.imshow(data_full[:,:,0].T, cmap=c_map, vmin=v_min, vmax=v_max, aspect='auto', extent=[0, (ntr-1)*dx, (ns-1)*dt*1e9, 0])
plt.title('High Frequency', fontsize=44, fontweight='bold')
plt.xlabel('Distance (m)', fontsize=40)
plt.ylabel('Time (ns)', fontsize=40)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)

ax2 = plt.subplot(412)
plt.imshow(img_3, cmap=c_map, vmin=v_min, vmax=v_max, aspect='auto', extent=[0, (ntr-1)*dx, (ns-1)*dt*1e9, 0])
plt.title('Low Frequency', fontsize=44, fontweight='bold')
plt.xlabel('Distance (m)', fontsize=40)
plt.ylabel('Time (ns)', fontsize=40)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)

ax3 = plt.subplot(413)
plt.imshow(data_full[:,:,1].T, cmap=c_map, vmin=v_min, vmax=v_max, aspect='auto', extent=[0, (ntr-1)*dx, (ns-1)*dt*1e9, 0])
plt.title('CycleGAN', fontsize=44, fontweight='bold')
plt.xlabel('Distance (m)', fontsize=40)
plt.ylabel('Time (ns)', fontsize=40)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)

ax4 = plt.subplot(414)
h = plt.imshow(y_pred.T, cmap=c_map, vmin=v_min, vmax=v_max, aspect='auto', extent=[0, (ntr-1)*dx, (ns-1)*dt*1e9, 0])
plt.title('Fusion result', fontsize=44, fontweight='bold')
plt.xlabel('Distance (m)', fontsize=40)
plt.ylabel('Time (ns)', fontsize=40)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)

plt.show()

# %%
# Save final fusion result
plt.imsave('fusion_result.png', y_pred.T, cmap=c_map, vmin=-5, vmax=5)

# %%
# %% Evaluation Metrics
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(img1, img3):
    """
    Calculate comprehensive fusion quality metrics
    Args:
        img1: Source high-frequency GPR image
        img3: Fused/Low-frequency image
    Returns:
        metrics: Dictionary containing all evaluation metrics
    """
    # Convert to grayscale if needed
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img3.shape) > 2:
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    # Standardization
    img1 = (img1 - np.mean(img1)) / np.std(img1)
    img3 = (img3 - np.mean(img3)) / np.std(img3)
    # Extract ROI (first 500 rows)
    img1_roi = img1[:500].astype(np.float32)
    img3_roi = img3[:500].astype(np.float32)

    metrics = {}
    
    # RMSE
    metrics["RMSE"] = np.sqrt(np.mean((img1_roi - img3_roi)**2))
    
    # SSIM and PSNR
    metrics["SSIM"] = ssim(img1_roi, img3_roi, data_range=255)
    metrics["PSNR"] = psnr(img1_roi, img3_roi, data_range=255)

    # Edge preservation index
    def edge_preservation(ori, fused):
        sobel_x = cv2.Sobel(ori, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(ori, cv2.CV_64F, 0, 1, ksize=3)
        edge_ori = np.sqrt(sobel_x**2 + sobel_y**2)
        
        sobel_x = cv2.Sobel(fused, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(fused, cv2.CV_64F, 0, 1, ksize=3)
        edge_fused = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return ssim(edge_ori, edge_fused, data_range=edge_ori.max()-edge_ori.min())

    metrics["Edge_SSIM"] = edge_preservation(img1_roi, img3_roi)

    # Spectral consistency
    def spectral_consistency(ori, fused):
        f_ori = np.fft.fft2(ori)
        f_fused = np.fft.fft2(fused)
        
        mag_ori = np.abs(np.fft.fftshift(f_ori))
        mag_fused = np.abs(np.fft.fftshift(f_fused))
        
        return ssim(mag_ori, mag_fused, data_range=mag_ori.max()-mag_ori.min())

    metrics["Spectral_SSIM"] = spectral_consistency(img1_roi, img3_roi)
    
    # Information entropy
    def calculate_entropy(image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.ravel()
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
        return entropy
    
    img1_uint8 = np.round(img1).astype(np.uint8)
    img3_uint8 = np.round(img3).astype(np.uint8)
    metrics["Entropy_Source"] = calculate_entropy(img1_uint8)
    metrics["Entropy_Fused"] = calculate_entropy(img3_uint8)
    metrics["Entropy_Delta"] = metrics["Entropy_Fused"] - metrics["Entropy_Source"]
    
    return metrics

# Evaluate fusion result
metrics_result = calculate_metrics(data_full[:,:,0].T, y_pred.T)

print("Fusion Quality Evaluation Results:")
print(f"SSIM: {metrics_result['SSIM']:.4f}")
print(f"PSNR: {metrics_result['PSNR']:.2f} dB")
print(f"RMSE: {metrics_result['RMSE']:.2f}")
print(f"Edge Preservation SSIM: {metrics_result['Edge_SSIM']:.4f}")
print(f"Spectral Consistency SSIM: {metrics_result['Spectral_SSIM']:.4f}")
print(f"Entropy (Source): {metrics_result['Entropy_Source']:.4f}")
print(f"Entropy (Fused): {metrics_result['Entropy_Fused']:.4f}")
print(f"Entropy Change: {metrics_result['Entropy_Delta']:.4f}")

# %%
# Evaluate Cycle-GAN generated result
metrics_result = calculate_metrics(data_full[:,:,0].T, data_full[:,:,1].T)

print("Cycle-GAN Generation Quality Evaluation Results:")
print(f"SSIM: {metrics_result['SSIM']:.4f}")
print(f"PSNR: {metrics_result['PSNR']:.2f} dB")
print(f"RMSE: {metrics_result['RMSE']:.2f}")
print(f"Edge Preservation SSIM: {metrics_result['Edge_SSIM']:.4f}")
print(f"Spectral Consistency SSIM: {metrics_result['Spectral_SSIM']:.4f}")
print(f"Entropy (Source): {metrics_result['Entropy_Source']:.4f}")
print(f"Entropy (Fused): {metrics_result['Entropy_Fused']:.4f}")
print(f"Entropy Change: {metrics_result['Entropy_Delta']:.4f}")

# %% 
# Evaluate Transformer fusion performance
metrics_result = calculate_metrics(data_full[:,:,0].T, img_3)

print("Transformer Fusion Quality Evaluation Results:")
print(f"SSIM: {metrics_result['SSIM']:.4f}")
print(f"PSNR: {metrics_result['PSNR']:.2f} dB")
print(f"RMSE: {metrics_result['RMSE']:.2f}")
print(f"Edge Preservation SSIM: {metrics_result['Edge_SSIM']:.4f}")
print(f"Spectral Consistency SSIM: {metrics_result['Spectral_SSIM']:.4f}")
print(f"Entropy (Source): {metrics_result['Entropy_Source']:.4f}")
print(f"Entropy (Fused): {metrics_result['Entropy_Fused']:.4f}")
print(f"Entropy Change: {metrics_result['Entropy_Delta']:.4f}")
# %%