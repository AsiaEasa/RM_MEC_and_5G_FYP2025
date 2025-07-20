import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('5ep_10ta_19_7_4.txt', mode='a'),
        logging.StreamHandler()
    ]
)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parameters
GAMMA = 0.99
LR = 0.001
MIN_BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPSILON = 0.1
EPSILON_MAX = 1.0
EPSILON_GROWTH = 1.001
EPSILON_MIN = 0.0
EPSILON_DECAY = 0.99
TARGET_UPDATE = 10
NUM_EPISODES = 5
NUM_DEVICES = 4
FEATURES_PER_DEVICE = 3
INPUT_DIM = (NUM_DEVICES, FEATURES_PER_DEVICE)
DESIRED_ACCURACY = 0.98
MAX_ITERATIONS = 1
ALPHA_N = 3.0
ALPHA_E = 2.0
ALPHA_L = 2.0
L_MAX = 500
G = 7000
TAU = 1e-28
MU_MB = 1
MU_BITS = MU_MB * 8 * 1e6
D = 20 * 1e6
LAMBDA = 1
ENERGY_SCALE = 1e12

# CNN for DDQN
class CNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Global Model for federated learning
class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Changed input channels to 1 for MNIST
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjusted for 28x28 MNIST images: (28/2)/2 = 7
        self.dropout = nn.Dropout(0.2)  # Adding Dropout with 0.2 probability
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply Dropout after fc1
        x = self.fc2(x)
        return x

# FiveGNetwork
class FiveGNetwork:
    def __init__(self, base_bandwidth=1000, base_latency=0.005, capacity=1000, spectrum="sub6"):
        self.base_bandwidth = base_bandwidth
        self.base_latency = base_latency
        self.capacity = capacity
        self.connected_devices = {}
        self.spectrum = spectrum
        self.network_slice = {
            "bandwidth": base_bandwidth,
            "latency": base_latency,
        }
        self.interference_factor = 0.01
        self.fading_factor = 0.97 if spectrum == "sub6" else 0.9
        self.packet_loss_rate = 0.0003
        self.throughput_history = []
        self.latency_history = []
        self.packet_loss_history = []

    def connect_device(self, device_id):
        if len(self.connected_devices) >= self.capacity:
            logging.info(f"5G Network: Capacity exceeded for device {device_id}, using fallback.")
            return 10, 0.01, self.packet_loss_rate

        self.connected_devices[device_id] = True
        slice_info = self.network_slice
        num_devices = len(self.connected_devices)
        interference = self.interference_factor * (num_devices - 1)
        variation = np.random.uniform(0.85, 1.15) * self.fading_factor
        effective_bandwidth = slice_info["bandwidth"] * (1 - interference) * variation
        effective_bandwidth = max(5, effective_bandwidth)
        latency = np.random.uniform(0.003, 0.02) * (1 + 0.01 * num_devices)
        latency = min(latency, 0.005)
        packet_loss = self.packet_loss_rate * (1 + np.random.uniform(0.02, 0.1) * num_devices)
        if self.spectrum == "mmWave":
            packet_loss *= 1.3
            packet_loss = min(packet_loss, 0.05)

        self.throughput_history.append(effective_bandwidth)
        self.latency_history.append(latency)
        self.packet_loss_history.append(packet_loss)
        
        logging.info(f"Device {device_id} connected: Bandwidth={effective_bandwidth:.2f} Mbps, "
                     f"Latency={latency*1000:.2f} ms, Packet Loss={packet_loss:.4f}, Service=mMTC")
        
        return effective_bandwidth, latency, packet_loss

    def disconnect_device(self, device_id):
        if device_id in self.connected_devices:
            del self.connected_devices[device_id]
            logging.info(f"Device {device_id} disconnected.")

    def reallocate_resources(self, selected_devices, devices):
        num_devices = len(self.connected_devices)
        num_active = len(selected_devices)
        if num_devices == 0:
            return
        active_bandwidth = self.base_bandwidth * 0.8 / max(1, num_active) if num_active > 0 else self.base_bandwidth
        inactive_bandwidth = self.base_bandwidth * 0.2 / max(1, num_devices - num_active) if num_devices > num_active else 0
        for device_id in self.connected_devices:
            original_bandwidth = self.network_slice["bandwidth"]
            self.network_slice["bandwidth"] = active_bandwidth if device_id in selected_devices else inactive_bandwidth
            effective_bandwidth, latency, packet_loss = self.connect_device(device_id)
            devices[device_id].update_network_params(effective_bandwidth, latency, packet_loss)
            self.network_slice["bandwidth"] = original_bandwidth
        logging.info(f"Resources reallocated: Active Bandwidth={active_bandwidth:.2f} Mbps, Inactive Bandwidth={inactive_bandwidth:.2f} Mbps")

    def simulate_channel_conditions(self):
        self.fading_factor = np.random.uniform(0.9, 0.98) if self.spectrum == "sub6" else np.random.uniform(0.85, 0.95)
        self.interference_factor = np.random.uniform(0.1, 0.005)
        logging.info(f"Channel updated: Fading={self.fading_factor:.2f}, Interference={self.interference_factor:.2f}")

    def get_network_status(self):
        return {
            "connected_devices": len(self.connected_devices),
            "average_throughput": np.mean(self.throughput_history) if self.throughput_history else 0,
            "average_latency": np.mean(self.latency_history) if self.latency_history else self.base_latency,
            "average_packet_loss": np.mean(self.packet_loss_history) if self.packet_loss_history else self.packet_loss_rate,
            "total_bandwidth": self.network_slice["bandwidth"]
        }

# EdgeDevice
class EdgeDevice:
    def __init__(self, id, cpu_freq, energy, bandwidth, fiveg_network):
        self.id = id
        self.cpu_freq = cpu_freq
        self.energy = energy
        self.base_bandwidth = bandwidth
        self.fiveg_network = fiveg_network
        self.model = GlobalModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()
        self.local_data = None
        self.effective_bandwidth, self.latency, self.packet_loss = self.fiveg_network.connect_device(self.id)

    def update_network_params(self, effective_bandwidth, latency, packet_loss):
        self.effective_bandwidth = effective_bandwidth
        self.latency = latency
        self.packet_loss = packet_loss
        logging.info(f"Device {self.id} updated: Bandwidth={self.effective_bandwidth:.2f} Mbps, "
                     f"Latency={self.latency*1000:.2f} ms, Packet Loss={self.packet_loss:.4f}")

    def set_local_data(self, data):
        if self.local_data is None:
            self.local_data = data
            logging.info(f"Device {self.id}: Assigned {len(data) if data else 0} samples")
        else:
            logging.info(f"Device {self.id}: Data already assigned, skipping")

    def charge_energy(self, w=1):
        return np.random.poisson(w) * ENERGY_SCALE

    def train_local_model(self, epochs, energy_rate=1):
        if not hasattr(self, 'call_counter'):
            self.call_counter = 0
        self.call_counter += 1

        if self.local_data is None:
            logging.info(f"Device {self.id}: No data assigned, skipping training.")
            return self.model.state_dict(), 0, 0, 0

        device = torch.device("cpu")
        self.model.to(device)
        self.model.train()

        batch_size = 1276
        start_idx = (self.call_counter - 1) * batch_size
        end_idx = start_idx + batch_size

        if self.call_counter == 12:
            self.call_counter = 0
            logging.info(f"End the training for device {self.id} and reset the call counter")

        indices = list(range(start_idx, min(end_idx, len(self.local_data))))
        logging.info(f"Device {self.id}, Call {self.call_counter}: Using indices {indices[:10]}{'...' if len(indices) > 10 else ''} (Total: {len(indices)} samples)")

        subset_data = torch.utils.data.Subset(self.local_data, indices)
        logging.info(f"Device {self.id}, Call {self.call_counter}: Training on {len(subset_data)} samples")
    
        loader = torch.utils.data.DataLoader(subset_data, batch_size=batch_size, shuffle=True, drop_last=False)

        for data, target in loader:
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

        T_local = MU_BITS * G / self.cpu_freq if self.cpu_freq > 0 else float('inf')
        B_k = (self.cpu_freq ** 2) * TAU * MU_BITS * G * ENERGY_SCALE
        C_k = self.charge_energy(energy_rate)
        logging.info(f"Device {self.id}: Charging with {C_k} energy units.")
        if self.energy < B_k:
            logging.info(f"Device {self.id}: Insufficient energy ({self.energy:.2f} < {B_k:.2f}), skipping training.")
            return self.model.state_dict(), 0, 0, 0
        self.energy = max(self.energy - B_k + C_k, 0)
        T_trans = D / self.effective_bandwidth if self.effective_bandwidth > 0 else float('inf')
        return self.model.state_dict(), T_local, T_trans, B_k

    def report_resources(self):
        return {'cpu_freq': self.cpu_freq, 'energy': self.energy, 'bandwidth': self.effective_bandwidth}

    def __del__(self):
        self.fiveg_network.disconnect_device(self.id)

# MECServer
class MECServer:
    def __init__(self, num_devices=NUM_DEVICES):
        self.fiveg_network = FiveGNetwork()
        self.global_model = GlobalModel()
        self.devices = []
        for i in range(num_devices):
            cpu_freq = np.random.uniform(0, 1)
            energy = np.random.uniform(2, 5)
            bandwidth = np.random.uniform(0, 2)
            device = EdgeDevice(i, cpu_freq, energy, bandwidth, self.fiveg_network)
            self.devices.append(device)
        self.energy_history = []
        self.bandwidth_history = []
        self.data_distributed = False
        logging.info(f"Created {len(self.devices)} unique devices")

    def distribute_data(self, trainset):
        if self.data_distributed:
            logging.info("Data already distributed, skipping")
            return
        logging.info(f"Starting data distribution for {len(trainset)} samples")
        data_size = len(trainset)
        data_per_device = data_size // len(self.devices)
        remainder = data_size % len(self.devices)
        indices = list(range(data_size))
        random.shuffle(indices)
        start_idx = 0
        for i, device in enumerate(self.devices):
            end_idx = start_idx + data_per_device + (1 if i < remainder else 0)
            device_indices = indices[start_idx:end_idx]
            subset = torch.utils.data.Subset(trainset, device_indices)
            device.set_local_data(subset)
            start_idx = end_idx
        logging.info(f"Distributed {data_size} samples across {len(self.devices)} devices")
        self.data_distributed = True

    def fed_avg(self, local_weights):
        avg_weights = {key: torch.zeros_like(local_weights[0][key]) for key in local_weights[0]}
        num_clients = len(local_weights)
        for weights in local_weights:
            for key in weights:
                avg_weights[key] += weights[key] / num_clients
        return avg_weights

    def simulate_training_round(self, selected_devices, epochs=10):
        local_weights = []
        total_energy = 0
        total_bandwidth = 0
        delays = []
        logging.info(f"Selected devices: {selected_devices}")
        self.fiveg_network.simulate_channel_conditions()
        self.fiveg_network.reallocate_resources(selected_devices, self.devices)

        for device_id in selected_devices:
            device = self.devices[device_id]
            total_bandwidth += device.effective_bandwidth
            weights, T_local, T_trans, energy_used = device.train_local_model(epochs)
            logging.info(f"Device {device_id}: Energy used: {energy_used} pJ, Bandwidth: {device.effective_bandwidth:.2f} Mbps")
            delay = T_local + T_trans
            delays.append(delay)
            if energy_used > 0:
                local_weights.append(weights)
                total_energy += energy_used

        if local_weights:
            new_weights = self.fed_avg(local_weights)
            self.global_model.load_state_dict(new_weights)
        self.energy_history.append(total_energy)
        self.bandwidth_history.append(total_bandwidth)
        max_latency = max(delays) if delays else 0
        return max_latency, total_energy, total_bandwidth, self.energy_history

    def evaluate_global_model(self, testloader):
        self.global_model.eval()
        correct = 0
        total = 0
        device = torch.device("cpu")
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

# DeviceSelectionEnv
class DeviceSelectionEnv:
    def __init__(self, mec_server):
        self.mec_server = mec_server
        self.num_devices = len(mec_server.devices)
        self.e_max_per_device = np.array([device.energy for device in mec_server.devices])        
        self.e_max_total = np.sum(self.e_max_per_device)
        self.action_space = [i for i in range(self.num_devices)]

    def generate_state(self):
        state = np.array([list(device.report_resources().values()) for device in mec_server.devices])
        return state

    def step(self, action):
        selected_indices = [i for i, val in enumerate(action) if val == 1]
        num_selected = len(selected_indices)

        max_latency, total_energy_consumed, total_bandwidth, e = self.mec_server.simulate_training_round(selected_indices)
        self.e_max_total = np.sum(e)

        reward = (ALPHA_N * (num_selected / NUM_DEVICES)
                  - ALPHA_E * (total_energy_consumed / self.e_max_total if self.e_max_total > 0 else 1.0)
                  - ALPHA_L * (max_latency / L_MAX))

        state = self.generate_state()
        logging.info(f"Round: Reward={reward:.2f}, Energy={total_energy_consumed}, Bandwidth={total_bandwidth:.2f}")
        return state, reward

    def reset(self):
        return self.generate_state()

# ReplayMemory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, min_batch_size):
        return random.sample(self.memory, min_batch_size)

    def __len__(self):
        return len(self.memory)

# DDQNAgent
class DDQNAgent:
    def __init__(self, input_channels, output_dim):
        self.policy_net = CNN(input_channels, output_dim)
        self.target_net = CNN(input_channels, output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON

    def preprocess_state(self, state):
        state_array = np.array(state).flatten()
        target_size = 256
        padded_state = np.pad(state_array, (0, target_size - len(state_array)), mode='constant', constant_values=0)
        return padded_state.reshape(1, 16, 16)

    def select_action(self, state):
        n = len(state)
        action = [0] * n
        if random.random() < self.epsilon:
            num_selected = 1
            selected_indices = random.sample(range(n), num_selected)
            for idx in selected_indices:
                action[idx] = 1
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(self.preprocess_state(state), dtype=torch.float32).unsqueeze(0)
                scores = self.policy_net(state_tensor).squeeze()
                with open("scores_output.txt", "a") as f:
                    f.write(f"Scores: {scores.tolist()}\n")
                num_selected = 1
                selected_indices = torch.argsort(scores, descending=True)[:num_selected].tolist()
                for idx in selected_indices:
                    action[idx] = 1
        return action

    def optimize_model(self):
        if len(self.memory) < MIN_BATCH_SIZE:
            return
        batch = self.memory.sample(MIN_BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(np.stack([self.preprocess_state(s) for s in states]), dtype=torch.float32)
        next_states = torch.tensor(np.stack([self.preprocess_state(s) for s in next_states]), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        q_values = self.policy_net(states).squeeze()
        next_q_values_policy = self.policy_net(next_states).detach().squeeze()
        next_q_values_target = self.target_net(next_states).detach().squeeze()

        expected_q_values = []
        predicted_q_values = []
        for i in range(MIN_BATCH_SIZE):
            action_indices = [idx for idx, val in enumerate(actions[i]) if val == 1]
            if not action_indices:
                predicted_q_values.append(torch.tensor(0.0, dtype=torch.float32))
                expected_q_values.append(rewards[i])
                continue
            q_pred = q_values[i, action_indices]
            predicted_q_values.append(q_pred.mean())
            q_next = next_q_values_target[i, torch.argmax(next_q_values_policy[i])]
            expected_q_values.append(rewards[i] + GAMMA * q_next)
        
        predicted_q_values = torch.stack(predicted_q_values)
        expected_q_values = torch.stack(expected_q_values)
        loss = nn.MSELoss()(predicted_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Main Simulation
if __name__ == "__main__":
    print("Starting program...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Changed to single channel for MNIST
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=85, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=85, shuffle=False, num_workers=2)

    mec_server = MECServer()
    mec_server.distribute_data(trainset)
    env = DeviceSelectionEnv(mec_server)
    agent = DDQNAgent(1, NUM_DEVICES)
    rewards_per_episode = []
    episode_energy_history = []
    episode_bandwidth_history = []
    episode_accuracies = []
    iteration_accuracies = []

    for episode in range(NUM_EPISODES):
        logging.info(f"\nStarting Episode {episode+1}")
        state = env.reset()
        total_reward = 0
        iteration = 0
        avg_reward = float('inf')
        accuracy = 0.0
        max_accuracy = 0.0
        episode_energy = 0
        episode_bandwidth = 0
        episode_iterations = 0
        

        while accuracy < DESIRED_ACCURACY:
            action = agent.select_action(state)
            next_state, reward = env.step(action)
            agent.memory.push((state, action, reward, next_state))
            agent.optimize_model()
            state = next_state
            total_reward += reward
            iteration += 1
            episode_iterations += 1
            episode_energy += mec_server.energy_history[-1] if mec_server.energy_history else 0
            episode_bandwidth += mec_server.bandwidth_history[-1] if mec_server.bandwidth_history else 0
            avg_reward = total_reward / iteration if iteration > 0 else reward

            accuracy = mec_server.evaluate_global_model(testloader)
            max_accuracy = max(max_accuracy, accuracy)
            iteration_accuracies.append(accuracy)
            logging.info(f"Iteration {iteration}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Accuracy: {accuracy:.4f}, max_accuracy: {max_accuracy:.2f}, Energy: {episode_energy:.2f}, Bandwidth: {episode_bandwidth:.2f}")
        agent.update_epsilon()
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        rewards_per_episode.append(total_reward)
        episode_accuracies.append(max_accuracy)
        episode_energy_history.append(episode_energy / episode_iterations if episode_iterations > 0 else 0)
        episode_bandwidth_history.append(episode_bandwidth / episode_iterations if episode_iterations > 0 else 0)
        logging.info(f"Episode {episode+1} Finished, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, "
                     f"Iterations: {iteration}, Max_Accuracy: {max_accuracy:.2f}, Energy: {episode_energy_history[-1]:.2f}, "
                     f"Bandwidth: {episode_bandwidth_history[-1]:.2f}")

    logging.info(f"Episode Accuracies so far: {episode_accuracies}")
    total_accuracy = np.mean(episode_accuracies) if episode_accuracies else 0
    logging.info(f"Total Accuracy across all episodes: {total_accuracy:.4f}")
    
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 4, 1)
    plt.plot(rewards_per_episode, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance")
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(episode_energy_history, label="Energy Consumption")
    plt.xlabel("Episode")
    plt.ylabel("Energy (pJ)")
    plt.title("Energy Consumption Over Episodes")
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(episode_bandwidth_history, label="Bandwidth Usage")
    plt.xlabel("Episode")
    plt.ylabel("Bandwidth (Mbps)")
    plt.title("Bandwidth Usage Over Episodes")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(iteration_accuracies, label="Accuracy per iteration")
    plt.xlabel("iteration")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Over Episodes")
    plt.legend()

    plt.tight_layout()
    plt.show() 
