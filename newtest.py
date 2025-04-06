import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import deque
import random
import pandas as pd

# Set random seeds for reproducibility
## This to set the same result
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 5G Network Class نشوف لو في مكاتب جاهزه للشبكه عشان نغوص فيها
class FiveGNetwork:
    def __init__(self, base_bandwidth=100, latency=0.001, capacity=10): # نكتب المتغيرات دي في البحث
        self.base_bandwidth = base_bandwidth  # Mbps (5G typically offers high bandwidth)
        self.latency = latency  # Seconds (low latency for 5G)
        self.capacity = capacity  # Max number of devices supported
        self.connected_devices = 0

    def connect_device(self):
        if self.connected_devices < self.capacity:
            self.connected_devices += 1
            # Dynamic bandwidth allocation (decreases slightly as more devices connect)
            effective_bandwidth = self.base_bandwidth * (1 - 0.05 * self.connected_devices) # نحدد  انو دا هو النطاق الترديي للجهاز بعد الاتصال بالشبكه ( دا النطاق الوفرتو شبكه الجيل الخامس)
            # نشوف ورقه تثبنت القانون دا 
            return effective_bandwidth, self.latency
        else:
            print("5G Network: Capacity exceeded, using fallback bandwidth.")
            return 10, 0.01  # Fallback to lower bandwidth and higher latency

    def disconnect_device(self):
        if self.connected_devices > 0:
            self.connected_devices -= 1

# Edge Device Class
class EdgeDevice:
    def __init__(self, id, cpu_freq, energy, bandwidth, fiveg_network):
        self.id = id
        self.cpu_freq = cpu_freq  # Hz
        self.energy = energy      # Energy units
        self.base_bandwidth = bandwidth  # Mbps (initial bandwidth)
        self.fiveg_network = fiveg_network  # Reference to 5G network
        self.model = CNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.local_data = None
        self.effective_bandwidth, self.latency = self.fiveg_network.connect_device()

    def set_local_data(self, data):
        self.local_data = data

    def charge_energy(self, w=1):
        Ck = np.random.poisson(w)
        return Ck

    def train_local_model(self, epochs, energy_rate=1):
        if self.local_data is None or len(self.local_data) < 85:
            print(f"Device {self.id}: Insufficient data ({len(self.local_data) if self.local_data else 0} samples), skipping training.")
            return self.model.state_dict(), 0, 0, 0
        
        device = torch.device("cpu")

        # Move the model to the CPU
        self.model.to(device)
        self.model.train()
 
        # The batch_size=85 refers to the number of data samples processed in one forward/backward pass during training.
        # It is not the size of the memory or the total dataset, but rather the size of the mini-batch used for training.
        # batch_size=85 To make μ = 1MB (1MB ≈ 85 CIFAR-10 images)
        loader = torch.utils.data.DataLoader(self.local_data, batch_size=85, shuffle=True, drop_last=True)

        # `epochs=1` means each device trains its local model for one full pass over its dataset before aggregation.
        # This reduces computation and energy usage on resource-constrained edge devices.
        for epoch in range(epochs): 
            for data, target in loader: # The data is a batch of images and target is the corresponding labels
                # data len = 85 . الجهاز بحتاج 147 حلقه عشان يخلص كل الداتا العندو
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            

        # Calculate computational time and energy consumption
        G = 7000 # Number of cpu cycles required to process one bit local data
        mu_MB = 1
        mu_bits = mu_MB * 8 * 1024 * 1024
        T_local = mu_bits * G / self.cpu_freq
        tau = 1e-28  # Time constant for energy consumption (in seconds)
        # Energy consumption (in Joules)
        B_k = (self.cpu_freq ** 2) * tau * mu_bits * G

        C_k = self.charge_energy(energy_rate)
        print(f"Device {self.id}: Charging with {C_k} energy units.")
        self.energy = max(self.energy - B_k + C_k, 0) # Ensure energy doesn't go negative

        # Check if energy is sufficient for training
        # If energy is less than B_k, skip training and return current model state
        if self.energy < B_k:
            print(f"Device {self.id}: Insufficient energy after update ({self.energy:.2f} < {B_k:.2f}), skipping training.")
            return self.model.state_dict(), 0, 0, 0

        # Transmission time using 5G network
        D = 20e6  # Model size in bits the size of the global model (20 MB)
        T_trans = D / self.effective_bandwidth  # Include latency from 5G
        
        return self.model.state_dict(), T_local, T_trans, B_k

    def report_resources(self):
        return {'cpu_freq': self.cpu_freq, 'energy': self.energy, 'bandwidth': self.effective_bandwidth}

    def __del__(self):
        self.fiveg_network.disconnect_device()

# MEC Server Class
class MECServer:
    def __init__(self, num_devices=4):
        self.fiveg_network = FiveGNetwork()
        self.global_model = CNN()
        self.devices = []
        for i in range(num_devices):
            cpu_freq = np.random.uniform(0, 1)  # Random CPU frequency
            energy = 5000 + (cpu_freq * 1000)   # Initial energy formula مفروض نكتبو في البحث انو فرضنا قانون وممكن نخليها راندوم لو ماعرفنا نثبتها
            # نشوف ورقه تثبنت القانون دا 
            bandwidth = np.random.uniform(0, 2) # نكتب ان دي النطاق الترددي للجهاز قبل الاتصال بالشبكه
            device = EdgeDevice(i, cpu_freq=cpu_freq, energy=energy,
                                bandwidth=bandwidth, fiveg_network=self.fiveg_network)
            self.devices.append(device)
        self.energy_history = []
        self.bandwidth_history = []
        
    # الفنكشن دا يشوفوه امنيه واسراء
    def distribute_data(self):
        data_per_device = len(trainset) // len(self.devices) 
        print(f"Distributing {data_per_device} samples per device")
        for i, device in enumerate(self.devices):
            subset = torch.utils.data.Subset(trainset, range(i * data_per_device, (i + 1) * data_per_device))
            device.set_local_data(subset)

    def fed_avg(self, local_weights, data_sizes):
        total_size = sum(data_sizes)
        avg_weights = {key: torch.zeros_like(local_weights[0][key]) for key in local_weights[0]}
        for weights, size in zip(local_weights, data_sizes):
            for key in weights:
                avg_weights[key] += weights[key] * (size / total_size)
        return avg_weights

    def simulate_training_round(self, selected_devices, epochs=1):
        local_weights = []
        data_sizes = []
        total_energy = 0
        total_bandwidth = 0
        
        for device_id in selected_devices:
            device = self.devices[device_id]
            weights, T_local, T_trans, energy_used = device.train_local_model(epochs)
            print(f"Device {device_id}: Local training completed. Energy used: {energy_used * 1e12:.10f} pJ, Bandwidth: {device.effective_bandwidth:.2f} Mbps")
            local_weights.append(weights)
            data_sizes.append(len(device.local_data))
            total_energy += energy_used
            total_bandwidth += device.effective_bandwidth
        
        new_weights = self.fed_avg(local_weights, data_sizes)
        self.global_model.load_state_dict(new_weights)

        self.energy_history.append(total_energy)
        self.bandwidth_history.append(total_bandwidth)

        return T_local + T_trans

    def evaluate_global_model(self):
        self.global_model.eval()
        correct = 0
        total = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(device)
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = correct / total
        return accuracy

# DDQN Class
class DDQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.update_target_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state): # Select an action a randomly with probability or a argmaxa AQ(sk a ) withprobability الفقره رقم 5
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            target_f = self.model(state)
            target_f[0, action] = target # Update the Q-value for the action taken
        
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Device Selection Function
def select_devices_ddqn(mec_server, ddqn, num_devices_to_select):
    state = np.array([list(d.report_resources().values()) for d in mec_server.devices]).flatten()
    state = state.reshape(1, -1)
    selected = []
    while len(selected) < num_devices_to_select: # مقروض شرط الحلقه يتغير بناء على حاله الجهاز وما يعتمد على عدد الاجهزه المحدده للتدريب في كل جوله
        action = ddqn.act(state)
        if action not in selected and action < len(mec_server.devices):
            selected.append(action)
    return selected

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False) # 32 is the batch size for testing ممكن نغيرها لي 85 لكن مابفرق عشان نحن هنا مركزين على اداء النموذج

# Main Simulation
if __name__ == "__main__":
    mec_server = MECServer(num_devices=4)
    for dev in mec_server.devices:
        print(f"Device {dev.id}: CPU={dev.cpu_freq:.4f}, Energy={dev.energy:.6f}, Base BW={dev.base_bandwidth:.4f}, Effective BW={dev.effective_bandwidth:.4f}")
    mec_server.distribute_data() # Distribute data to devices
    ddqn = DDQN(state_size=12, action_size=4) # 4 devices, 3 features (cpu_freq, energy, bandwidth)
    
    num_rounds = 5
    for round_num in range(num_rounds):
        selected_devices = select_devices_ddqn(mec_server, ddqn, num_devices_to_select=2)
        delay = mec_server.simulate_training_round(selected_devices, epochs=1) # 1 epoch for local training 
        accuracy = mec_server.evaluate_global_model()
        print(f"Round {round_num+1}: Accuracy = {accuracy:.4f}, Delay = {delay:.4f}")
        total_energy = mec_server.energy_history[-1]
        total_bandwidth = mec_server.bandwidth_history[-1]
        print(f"📊 Round {round_num+1}: Total energy used = {total_energy * 1e12:.10f} pJ, Total bandwidth used = {total_bandwidth:.4f} Mbps")

        reward = accuracy - delay / 1000
        state = np.array([list(d.report_resources().values()) for d in mec_server.devices]).flatten().reshape(1, -1)
        next_state = state
        ddqn.remember(state, selected_devices[0], reward, next_state, False)
        
        if len(ddqn.memory) > 32:
            ddqn.replay(32)
        ddqn.update_target_model()
        print("🔄 Updating target network...")

    final_accuracy = mec_server.evaluate_global_model()
    print(f"🏁 Final accuracy after {num_rounds} rounds: {final_accuracy:.4f}")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mec_server.energy_history, label='Energy Consumption')
    plt.xlabel('Training Round')
    plt.ylabel('Energy Units')
    plt.title('Energy Consumption Over Rounds')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mec_server.bandwidth_history, label='Bandwidth Usage')
    plt.xlabel('Training Round')
    plt.ylabel('Bandwidth (Mbps)')
    plt.title('Bandwidth Usage Over Rounds')
    plt.legend()

    plt.tight_layout()
    plt.show()
