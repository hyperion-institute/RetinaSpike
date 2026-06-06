import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

"""***** Config *****"""
TIMESTEP = 12
THRESHOLD = 0.4

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

EXPERIMENT_NAME = "cossimilarity_contrast_encode"

"""***** Data *****"""
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""***** Feature Extractor *****"""
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 14 -> 7
        )

    def forward(self, x):
        x = self.features(x)
        norm = torch.norm(x, dim=1, keepdim=True) + 1e-8
        x = x / norm
        return x

"""***** Cosine similarity / Contrast *****"""
def compute_contrast(f):
    B, C, H, W = f.shape
    patches = F.unfold(f, kernel_size=3, padding=1)
    patches = patches.view(B, C, 9, H, W)
    center = f.unsqueeze(2)
    sim = (center * patches).sum(dim=1)
    contrast = (1 - sim).sum(dim=1)
    return contrast

"""***** Encoding to Spike *****"""
def encode_to_spike(contrast, T=TIMESTEP, threshold=THRESHOLD):
    B, H, W = contrast.shape

    c_min = contrast.amin(dim=(1,2), keepdim=True)
    c_max = contrast.amax(dim=(1,2), keepdim=True)
    c_norm = (contrast - c_min) / (c_max - c_min + 1e-8)

    thresh = torch.quantile(c_norm.view(B, -1), threshold, dim=1)
    thresh = thresh.view(B, 1, 1)
    mask = c_norm > thresh

    t = (1 - c_norm) * (T - 1)
    t_floor = torch.floor(t).long()
    t_ceil = torch.clamp(t_floor + 1, max=T - 1)
    alpha = t - t_floor.float()

    spike = torch.zeros((T, B, 1, H, W), device=contrast.device)
    b_idx, h_idx, w_idx = torch.nonzero(mask, as_tuple=True)

    tf = t_floor[b_idx, h_idx, w_idx]
    tc = t_ceil[b_idx, h_idx, w_idx]
    a = alpha[b_idx, h_idx, w_idx]

    spike[tf, b_idx, 0, h_idx, w_idx] += (1 - a)
    spike[tc, b_idx, 0, h_idx, w_idx] += a

    return spike

"""***** SNN model *****"""
class SNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 7 * 7, 10)

        spike_grad = snn.surrogate.fast_sigmoid()

        self.lif1 = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=spike_grad)

    def forward(self, spike):
        T, B, C, H, W = spike.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        out_sum = torch.zeros(B, 10, device=spike.device)
        snn_internal_spikes = 0.0
        
        for t in range(T):
            x = spike[t]

            # conv1
            x = self.conv1(x)
            spk1, mem1 = self.lif1(x, mem1)
            snn_internal_spikes += spk1.sum().item()

            # conv2
            x = self.conv2(spk1)
            spk2, mem2 = self.lif2(x, mem2)
            snn_internal_spikes += spk2.sum().item()

            # flatten & fc
            x = spk2.view(B, -1)
            x = self.fc(x)
            spk3, mem3 = self.lif3(x, mem3)
            snn_internal_spikes += spk3.sum().item()

            out_sum += mem3

        return out_sum / T, snn_internal_spikes

"""***** Initialize model  *****"""
feature_net = FeatureExtractor().to(device)
snn_model = SNNModel().to(device)

optimizer = torch.optim.Adam(
    list(feature_net.parameters()) + list(snn_model.parameters()),
    lr=LEARNING_RATE
)

criterion = nn.CrossEntropyLoss()

"""***** Train *****"""
def train():
    feature_net.train()
    snn_model.train()
    running_loss = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        f = feature_net(data)
        contrast = compute_contrast(f)
        spike = encode_to_spike(contrast, T=TIMESTEP, threshold=THRESHOLD)

        output, _ = snn_model(spike)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss

"""***** Evaluate *****"""
def evaluate_model():
    feature_net.eval()
    snn_model.eval()
    correct = 0
    total = 0

    total_input_spikes = 0
    total_snn_spikes = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            f = feature_net(data)
            contrast = compute_contrast(f)
            spike = encode_to_spike(contrast, T=TIMESTEP, threshold=THRESHOLD)

            total_input_spikes += (spike > 1e-5).float().sum().item()

            output, active_internal_spikes = snn_model(spike)
            total_snn_spikes += active_internal_spikes

            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    acc = 100.0 * correct / total
    return acc, total_input_spikes, total_snn_spikes

"""***** Run and save best *****"""
if __name__ == "__main__":
    best_acc = 0
    
    best_report = {
        "T": TIMESTEP,
        "Thresh": THRESHOLD,
        "Acc": 0.0,
        "In_Spk_Per_Img": 0.0,
        "SNN_Spk_Per_Img": 0.0,
        "Sparsity": 0.0
    }

    print("\n" + "="*70)
    print(f"STARTING SINGLE CONFIGURATION TRAIN & BENCHMARK")
    print(f"TIMESTEP (T) = {TIMESTEP} | THRESHOLD = {THRESHOLD}")
    print("="*70)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1:02d}/{EPOCHS:02d}")

        loss = train()
        acc, in_spks, snn_spks = evaluate_model()

        total_neurons_per_step = (1 * 7 * 7) + (32 * 7 * 7) + (64 * 7 * 7) + 10
        max_possible_system_spikes = 10000 * TIMESTEP * total_neurons_per_step

        total_actual_spikes = in_spks + snn_spks
        firing_rate = (total_actual_spikes / max_possible_system_spikes) * 100
        sparsity = 100.0 - firing_rate

        print(f"   Train Loss: {loss:.4f}")
        print(f"   Accuracy  : {acc:.2f}% | Input Spikes: {int(in_spks):,} | SNN Spikes: {int(snn_spks):,}")
        print(f"   Sparsity  : {sparsity:.2f}%")

        if acc > best_acc:
            best_acc = acc

            best_report = {
                "T": TIMESTEP,
                "Thresh": THRESHOLD,
                "Acc": round(best_acc, 2),
                "In_Spk_Per_Img": round(in_spks / 10000, 1),
                "SNN_Spk_Per_Img": round(snn_spks / 10000, 1),
                "Sparsity": round(sparsity, 2)
            }

            torch.save({
                "feature_net": feature_net.state_dict(),
                "snn_model": snn_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "epoch": epoch,
                "T": TIMESTEP,
                "threshold": THRESHOLD
            }, f"{EXPERIMENT_NAME}.pth")

            print("   >>> BEST MODEL SAVED")

        print(f"Best Accuracy So Far: {best_acc:.2f}%")

    """***** Load best and report *****""" 
    try:
        checkpoint = torch.load(f"{EXPERIMENT_NAME}.pth")
        feature_net.load_state_dict(checkpoint["feature_net"])
        snn_model.load_state_dict(checkpoint["snn_model"])
        saved_acc_msg = f"Saved Best Accuracy: {checkpoint['best_acc']:.2f}% (At Epoch {checkpoint['epoch'] + 1})"
    except FileNotFoundError:
        saved_acc_msg = "No checkpoint file found (Model didn't improve during training)."

    print("\n" + "="*70)
    print("FINAL BENCHMARK REPORT (BEST MODEL METRICS)")
    print("="*70)
    print(f"Timestep (T)            : {best_report['T']}")
    print(f"Threshold               : {best_report['Thresh']}")
    print(f"Max Accuracy (%)        : {best_report['Acc']}%")
    print(f"Input Spikes (Per Image): {best_report['In_Spk_Per_Img']}")
    print(f"SNN Spikes (Per Image)  : {best_report['SNN_Spk_Per_Img']}")
    print(f"System Sparsity (%)     : {best_report['Sparsity']}%")
    print("="*70)
    print(saved_acc_msg)
