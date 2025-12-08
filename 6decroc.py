#  \\\\\\\[|end]
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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import csv
from collections import Counter
import matplotlib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import sys
matplotlib.use('Agg')

# Setup logging with UTF-8 encoding to handle Arabic characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('6dec-output.txt', mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
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
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
NUM_EPISODES = 1000
NUM_DEVICES = 4
FEATURES_PER_DEVICE = 3
INPUT_DIM = (NUM_DEVICES, FEATURES_PER_DEVICE)
DESIRED_ACCURACY = 0.86
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
SAMPLES_PER_DEVICE = 1276  # عدد العينات لكل جهاز (يجب أن يكون مضاعفًا لعدد الفئات)
SAMPLES_PER_CLASS = SAMPLES_PER_DEVICE // 10  # عدد العينات لكل فئة (1 عينة لكل فئة)

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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.2)
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
        x = self.dropout(x)
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
            logging.info(f"5G Network: capacity exceeded for device {device_id}, using fallback.")
            return 10, 0.01, self.packet_loss_rate

        self.connected_devices[device_id] = True
        slice_info = self.network_slice
        num_devices = len(self.connected_devices)
        interference = self.interference_factor * (num_devices - 1)
        variation = np.random.uniform(0.95, 1.05) * self.fading_factor
        effective_bandwidth = slice_info["bandwidth"] * (1 - interference) * variation
        effective_bandwidth = max(5, effective_bandwidth)
        latency = np.random.uniform(0.003, 0.01) * (1 + 0.005 * num_devices)  # قلل الضرب من 0.01 إلى 0.005
        latency = min(latency, 0.005)
        packet_loss = self.packet_loss_rate * (1 + np.random.uniform(0.01, 0.05) * num_devices)  # قلل النطاق
        if self.spectrum == "mmWave":
            packet_loss *= 1.3
            packet_loss = min(packet_loss, 0.05)

        self.throughput_history.append(effective_bandwidth)
        self.latency_history.append(latency)
        self.packet_loss_history.append(packet_loss)
        
        logging.info(f"Device {device_id} connected: bandwidth={effective_bandwidth:.2f} Mbps, "
                     f"latency={latency*1000:.2f} ms, packet_loss={packet_loss:.4f}, service=mMTC")

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
        inactive_bandwidth = max(5, self.base_bandwidth * 0.2 / max(1, num_devices - num_active))  # حد أدنى 5 Mbps
        for device_id in self.connected_devices:
            original_bandwidth = self.network_slice["bandwidth"]
            self.network_slice["bandwidth"] = active_bandwidth if device_id in selected_devices else inactive_bandwidth
            effective_bandwidth, latency, packet_loss = self.connect_device(device_id)
            devices[device_id].update_network_params(effective_bandwidth, latency, packet_loss)
            self.network_slice["bandwidth"] = original_bandwidth
        logging.info(f"Resource reallocation: active_bandwidth={active_bandwidth:.2f} Mbps, inactive_bandwidth={inactive_bandwidth:.2f} Mbps")

    def simulate_channel_conditions(self):
        self.fading_factor = np.random.uniform(0.9, 0.98) if self.spectrum == "sub6" else np.random.uniform(0.85, 0.95)
        self.interference_factor = np.random.uniform(0.005, 0.01)
        logging.info(f"Channel conditions updated: fading={self.fading_factor:.2f}, interference={self.interference_factor:.2f}")

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
        logging.info(f"Device {self.id} updated: bandwidth={self.effective_bandwidth:.2f} Mbps, "
                     f"latency={self.latency*1000:.2f} ms, packet_loss={self.packet_loss:.4f}")

    def set_local_data(self, data):
        self.local_data = data  # استبدال البيانات القديمة مباشرة
        logging.info(f"Device {self.id}: assigned {len(data) if data else 0} samples")

    def charge_energy(self, w=1):
        return np.random.poisson(w* 0.8) * ENERGY_SCALE

    def train_local_model(self, epochs, energy_rate=1):
        if not hasattr(self, 'call_counter'):
            self.call_counter = 0
        self.call_counter += 1

        if self.local_data is None:
            logging.info(f"Device {self.id}: no assigned data, skipping training.")
            return self.model.state_dict(), 0, 0, 0

        device = torch.device("cpu")
        self.model.to(device)
        self.model.train()

        batch_size = 1276
        
    
        loader = torch.utils.data.DataLoader(self.local_data, batch_size=batch_size, shuffle=True, drop_last=False)

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
        logging.info(f"Device {self.id}: charged with {C_k} energy units.")
        if self.energy < B_k:
            logging.info(f"Device {self.id}: insufficient energy ({self.energy:.2f} < {B_k:.2f}), skipping training.")
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
        self.class_indices = {i: [] for i in range(10)}  # قوائم المؤشرات لكل فئة
        self.class_pointers = {i: 0 for i in range(10)}  # مؤشرات لتتبع العينات الموزعة لكل فئة
        self.distributed_indices = []  # لتتبع جميع المؤشرات الموزعة
        logging.info(f"Created {len(self.devices)} unique devices")

    def initialize_class_indices(self, trainset):
        for idx in range(len(trainset)):
            _, label = trainset[idx]
            self.class_indices[label].append(idx)
        for label in self.class_indices:
        # لا نقوم بخلط المؤشرات للحفاظ على الترتيب "الأول"
            logging.info(f"Class {label}: {len(self.class_indices[label])} samples")

    def distribute_data(self, trainset, selected_devices):
        if not self.class_indices[0]:
            self.initialize_class_indices(trainset)

        for device_id in selected_devices:
            device = self.devices[device_id]
            device_indices = []
    
            for label in range(10):
                class_list = self.class_indices[label]
                class_len = len(class_list)
                if class_len == 0:
                    logging.warning(f"Class {label}: no samples available")
                    continue

                pointer = self.class_pointers[label]
                num_samples = SAMPLES_PER_CLASS

                selected_indices = []
                if pointer + num_samples > class_len:
                    selected_indices.extend(class_list[pointer:])
                    remaining = num_samples - (class_len - pointer)
                    selected_indices.extend(class_list[:remaining])
                    new_pointer = remaining
                else:
                    selected_indices.extend(class_list[pointer : pointer + num_samples])
                    new_pointer = pointer + num_samples

                self.class_pointers[label] = new_pointer % class_len
                device_indices.extend(selected_indices)

            if device_indices:
                subset = torch.utils.data.Subset(trainset, device_indices)
                device.set_local_data(subset)
                labels_for_device = [trainset[idx][1] for idx in device_indices]
                label_counts = dict(Counter(labels_for_device))

                logging.info(f" Device {device_id} received {len(labels_for_device)} samples")
                for cls, count in sorted(label_counts.items()):
                    logging.info(f"   - Class {cls}: {count} samples")
                logging.debug(f"   Indices: {device_indices}")
            else:
                logging.warning(f" No data distributed to device {device_id}")

    def fed_avg(self, local_weights):
        avg_weights = {key: torch.zeros_like(local_weights[0][key]) for key in local_weights[0]}
        num_clients = len(local_weights)
        for weights in local_weights:
            for key in weights:
                avg_weights[key] += weights[key] / num_clients
        return avg_weights

    def simulate_training_round(self, selected_devices, epochs=10, episode=0):
        np.random.seed(SEED + episode)  # تثبيت seed في بداية الدالة
        random.seed(SEED + episode)
        local_weights = []
        total_energy = 0
        total_bandwidth = 0
        delays = []
        logging.info(f"Selected devices: {selected_devices}")
        self.fiveg_network.simulate_channel_conditions()
        self.fiveg_network.reallocate_resources(selected_devices, self.devices)

        # توزيع البيانات على الأجهزة المختارة
        self.distribute_data(trainset, selected_devices)

        for device_id in selected_devices:
            device = self.devices[device_id]
            total_bandwidth += device.effective_bandwidth
            weights, T_local, T_trans, energy_used = device.train_local_model(epochs)
            logging.info(f"Device {device_id}: energy used: {energy_used} J, bandwidth: {device.effective_bandwidth:.2f} Mbps")
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
        all_preds = []
        all_targets = []
        all_probs = []  # store probability vectors for ROC/AUC
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                outputs = self.global_model(data)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        accuracy = correct / total if total > 0 else 0
        return accuracy, np.array(all_preds), np.array(all_targets), np.array(all_probs)


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

    def step(self, action, episode=0):
        selected_indices = [i for i, val in enumerate(action) if val == 1]
        num_selected = len(selected_indices)

        max_latency, total_energy_consumed, total_bandwidth, e = self.mec_server.simulate_training_round(selected_indices, episode=episode)
        self.e_max_total = np.sum(e)

        reward = (ALPHA_N * (num_selected / NUM_DEVICES)
                  - ALPHA_E * (total_energy_consumed / self.e_max_total if self.e_max_total > 0 else 1.0)
                  - ALPHA_L * (max_latency / L_MAX))

        state = self.generate_state()
        logging.info(f"Round: reward={reward:.2f}, energy={total_energy_consumed}, bandwidth={total_bandwidth:.2f}")
        return state, reward, max_latency

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
        state_array = (state_array - np.mean(state_array)) / (np.std(state_array) + 1e-8)  # تطبيع
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
                with open("scores_output_fixed_5g_lat6dec.txt", "a") as f:
                    f.write(f"scores: {scores.tolist()}\n")
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

# Function to plot confusion matrix
def plot_confusion_matrix(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

# Function to plot ROC curves (similar to confusion matrix style)
def plot_roc_curves(all_targets, all_probs, classes):
    plt.figure(figsize=(10, 8))
    n_classes = len(classes)
    y_true_bin = label_binarize(all_targets, classes=list(range(n_classes)))
    # compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves per class')
    plt.legend(loc="lower right")
    plt.grid(True)

# Main Simulation
if __name__ == "__main__":
    print("Starting program...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=85, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=85, shuffle=False, num_workers=2)

    mec_server = MECServer()
    env = DeviceSelectionEnv(mec_server)
    agent = DDQNAgent(1, NUM_DEVICES)
    rewards_per_episode = []
    episode_energy_history = []
    episode_bandwidth_history = []
    episode_accuracies = []
    iteration_accuracies = []
    episode_latency_history = []
    all_preds_list = []  # لتخزين التنبؤات لكل تكرار
    all_targets_list = []  # لتخزين الأهداف لكل تكرار
    all_probs_list = []  # لتخزين الاحتمالات لكل تكرار لـ ROC
    episode_precisions = []
    episode_recalls = []
    episode_f1s = []
    episode_aucs = []
    

    # Create CSV file and write header
    with open("episode_metrics.csv", "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Precision", "Recall", "F1-score", "AUC"])

    for episode in range(NUM_EPISODES):
        logging.info(f"\nStarting episode {episode+1}")
        state = env.reset()
        total_reward = 0
        iteration = 0
        avg_reward = float('inf')
        accuracy = 0.0
        max_accuracy = 0.0
        episode_energy = 0
        episode_bandwidth = 0
        episode_iterations = 0
        episode_latency_list = []

        
        while accuracy < DESIRED_ACCURACY:
            action = agent.select_action(state)
            next_state, reward, max_latency = env.step(action, episode)
            agent.memory.push((state, action, reward, next_state))
            agent.optimize_model()
            state = next_state
            total_reward += reward
            iteration += 1
            episode_iterations += 1
            episode_energy += mec_server.energy_history[-1] if mec_server.energy_history else 0
            episode_bandwidth += mec_server.bandwidth_history[-1] if mec_server.bandwidth_history else 0
            avg_reward = total_reward / iteration if iteration > 0 else reward

            accuracy, all_preds, all_targets, all_probs = mec_server.evaluate_global_model(testloader)

            # حساب المقاييس الأساسية (macro averages مفيدة للـ multiclass)
            precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
             # ROC / AUC — تحويل الـ targets إلى one-hot (binarize) لمتعدد الفئات
            n_classes = all_probs.shape[1] if all_probs.ndim > 1 else 1
            y_true_bin = label_binarize(all_targets, classes=list(range(n_classes)))

            # roc_auc_score for multiclass: use 'ovr' or 'ovo' (one-vs-rest typical)
            try:
                auc_macro = roc_auc_score(y_true_bin, all_probs, average='macro', multi_class='ovr')
            except Exception as e:
                logging.warning(f"ROC AUC failed: {e}")
                auc_macro = None
            episode_precisions.append(precision)
            episode_recalls.append(recall)
            episode_f1s.append(f1)
            episode_aucs.append(auc_macro if auc_macro is not None else 0)


            # Log/print
            logging.info(f"Accuracy: {accuracy:.4f}  Precision(macro): {precision:.4f}  Recall(macro): {recall:.4f}  F1(macro): {f1:.4f}")
            print(f"Acc={accuracy:.4f} Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")

           

            if auc_macro is not None:
                logging.info(f"AUC (macro, OVR): {auc_macro:.4f}")
                print(f"AUC (macro): {auc_macro:.4f}")
            

            max_accuracy = max(max_accuracy, accuracy)
            iteration_accuracies.append(accuracy)
            all_preds_list.append(all_preds)
            all_targets_list.append(all_targets)
            all_probs_list.append(all_probs)
            episode_latency_list.append(max_latency)
            logging.info(f"Iteration {iteration}, total_reward: {total_reward:.2f}, avg_reward: {avg_reward:.2f}, "
                        f"accuracy: {accuracy:.4f}, max_accuracy: {max_accuracy:.2f}, energy: {episode_energy:.2f}, "
                        f"bandwidth: {episode_bandwidth:.2f}")

        agent.update_epsilon()
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        rewards_per_episode.append(total_reward)
        episode_accuracies.append(max_accuracy)
        episode_energy_history.append(episode_energy / episode_iterations if episode_iterations > 0 else 0)
        episode_bandwidth_history.append(episode_bandwidth / episode_iterations if episode_iterations > 0 else 0)
        episode_latency_history.append(np.mean(episode_latency_list) if episode_latency_list else 0)
        # Save metrics for this episode to CSV
        with open("episode_metrics.csv", "a", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode + 1,
                episode_precisions[-1],
                episode_recalls[-1],
                episode_f1s[-1],
                episode_aucs[-1]
            ])

        logging.info(f"Finished episode {episode+1}, total_reward: {total_reward:.2f}, avg_reward: {avg_reward:.2f}, "
                    f"iterations: {iteration}, max_accuracy: {max_accuracy:.2f}, energy: {episode_energy_history[-1]:.2f}, "
                    f"bandwidth: {episode_bandwidth_history[-1]:.2f}")
        
    logging.info(f"Episode accuracies so far: {episode_accuracies}")
    total_accuracy = np.mean(episode_accuracies) if episode_accuracies else 0
    logging.info(f"Overall accuracy across episodes: {total_accuracy:.4f}")

    # رسم مصفوفة الالتباس باستخدام التنبؤات من التكرار 160
    if len(all_preds_list) >= 160 and len(all_targets_list) >= 160:
        classes = [str(i) for i in range(10)]  # فئات MNIST (0-9)
        plot_confusion_matrix(all_targets_list[159], all_preds_list[159], classes)
        plt.savefig('confusion_matrix_fixed_5g_lat_6dec.png')  # حفظ مصفوفة الالتباس
        plt.close()
        
        # رسم منحنيات ROC بنفس الطريقة، باستخدام البيانات من التكرار 160
        plot_roc_curves(all_targets_list[159], all_probs_list[159], classes)
        plt.savefig('roc_curves_fixed_5g_lat_6dec.png')  # حفظ منحنيات ROC
        plt.close()
    else:
        logging.warning("Cannot plot confusion matrix or ROC curves: fewer than 160 iterations completed")

    # رسم المقاييس الأخرى
    plt.figure(figsize=(10, 6))
    plt.plot(episode_energy_history)
    plt.xlabel("Episode")
    plt.ylabel("Energy (unit)")
    plt.title("Energy Consumption")
    plt.legend()
    plt.savefig('energy per episode_fixed_5g_lat_6dec.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(episode_latency_history)
    plt.xlabel("Episode")
    plt.ylabel("Latency (s)")
    plt.title("Latency per Episode")
    plt.legend()
    plt.savefig('Latency per episode_fixed_5g_lat_6dec.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(iteration_accuracies, label="Accuracy per iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Model accuracy across episodes")
    plt.legend()
    plt.savefig('Accuracy per iteration1_fixed_5g_lat_6dec.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(iteration_accuracies[:160])
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Model accuracy during first 160 iterations")
    plt.legend()
    plt.savefig('Accuracy per for 160 iteration1_fixed_5g_lat_v6dec.png')
    plt.close()
    



  
    
