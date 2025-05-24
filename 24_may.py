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

# إعداد logging لتسجيل الرسائل في ملف
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation_log_22th_5.txt', mode='a'),
        logging.StreamHandler()  # ملف لتسجيل الرسائل
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
# LR = .0001 , lr = .001 , NUM_EPISODES = 35 , epoch = 10
#  حساب الدقه الكليه
# تعديل متغيرات شبكه الجيل الخامس

GAMMA = 0.99
LR = 0.0001
MIN_BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPSILON = 0.1
EPSILON_MAX = 1.0
EPSILON_GROWTH = 1.001
EPSILON_MIN = 0.0
EPSILON_DECAY = 0.999
TARGET_UPDATE = 1
NUM_EPISODES = 35
NUM_DEVICES = 4
FEATURES_PER_DEVICE = 3
INPUT_DIM = (NUM_DEVICES, FEATURES_PER_DEVICE)
DESIRED_ACCURACY = 0.8
MAX_ITERATIONS = 5
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
# 64 X 64 X 64 
class CNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)  # من الطبقة الطيّ إلى 64 وحدة
        self.fc2 = nn.Linear(64, 64)          # 64 إلى 64 وحدة
        self.fc3 = nn.Linear(64, 64)          # 64 إلى 64 وحدة
        self.fc4 = nn.Linear(64, output_dim)  # من 64 إلى فضاء الإجراء

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # تسطيح الإدخال
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # بدون تنشيط في طبقة الإخراج
        return x.squeeze()  # Ensure output is (batch_size, output_dim)

# Global Model for federated learning
class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        # tells PyTorch to automatically calculate the size of the second dimension so that the
        # total number of elements in the tensor remains the same.
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# FiveGNetwork
import numpy as np

# نشوف مصدر للقوانيين والارقام
class FiveGNetwork:
    def __init__(self, base_bandwidth=1000, base_latency=0.005, capacity=1000, spectrum="sub6"):
        self.base_bandwidth = base_bandwidth  # Mbps (مناسب لـ mMTC)
        self.base_latency = base_latency  # seconds (منخفض لدعم mMTC)
        self.capacity = capacity  # دعم عدد كبير من الأجهزة
        self.connected_devices = {}  # Dictionary: device_id -> connected
        self.spectrum = spectrum  # "mmWave" or "sub6"
        self.network_slice = {
            "bandwidth": base_bandwidth,
            "latency": base_latency,
        }
        self.interference_factor = 0.01  # تأثير التداخل لكل جهاز
        self.fading_factor = 0.97 if spectrum == "sub6" else 0.9  # تأثير التلاشي
        self.packet_loss_rate = 0.0003  # معدل فقدان الحزم منخفض
        self.throughput_history = []
        self.latency_history = []
        self.packet_loss_history = []

    def connect_device(self, device_id):
        """Connect a device to the 5G network with mMTC service."""
        if len(self.connected_devices) >= self.capacity:
            logging.info(f"5G Network: Capacity exceeded for device {device_id}, using fallback.")
            return 10, 0.01, self.packet_loss_rate

        self.connected_devices[device_id] = True
        slice_info = self.network_slice
        num_devices = len(self.connected_devices)
        
        # حساب التداخل والتلاشي
        interference = self.interference_factor * (num_devices - 1)
        variation = np.random.uniform(0.9, 1.1) * self.fading_factor
        effective_bandwidth = slice_info["bandwidth"] * (1 - interference) * variation
        effective_bandwidth = max(5, effective_bandwidth)  # ضمان الحد الأدنى للنطاق الترددي
        
        # تعديل التأخير بناءً على ازدحام الشبكة
        latency = slice_info["latency"] * (1 + 0.05 * num_devices)
        latency = min(latency, 0.005)  # تأخير مناسب لـ mMTC (~5ms)
        
        # تعديل معدل فقدان الحزم بناءً على الطيف وازدحام الشبكة
        packet_loss = self.packet_loss_rate * (1 + 0.03 * num_devices)
        if self.spectrum == "mmWave":
            packet_loss *= 1.2  # زيادة طفيفة في mmWave

        # تحديث سجلات الأداء
        self.throughput_history.append(effective_bandwidth)
        self.latency_history.append(latency)
        self.packet_loss_history.append(packet_loss)
        
        logging.info(f"Device {device_id} connected: Bandwidth={effective_bandwidth:.2f} Mbps, "
                     f"Latency={latency*1000:.2f} ms, Packet Loss={packet_loss:.4f}, Service=mMTC")
        
        return effective_bandwidth, latency, packet_loss

    def disconnect_device(self, device_id):
        """Disconnect a device from the network."""
        if device_id in self.connected_devices:
            del self.connected_devices[device_id]
            logging.info(f"Device {device_id} disconnected.")

    # تعديل: دالة reallocate_resources المعدلة لتخصيص النطاق الترددي ديناميكيًا بناءً على الأجهزة النشطة
    def reallocate_resources(self, selected_devices, devices):
        """Reallocate bandwidth dynamically based on connected and active devices."""
        num_devices = len(self.connected_devices)
        num_active = len(selected_devices)
        if num_devices == 0:
            return
        
        # تخصيص 80% من النطاق الترددي للأجهزة النشطة و20% للأجهزة غير النشطة
        active_bandwidth = self.base_bandwidth * 0.8 / max(1, num_active) if num_active > 0 else self.base_bandwidth
        inactive_bandwidth = self.base_bandwidth * 0.2 / max(1, num_devices - num_active) if num_devices > num_active else 0
        
        # تحديث network_slice مؤقتًا لكل جهاز وإعادة اتصاله لتحديث effective_bandwidth
        for device_id in self.connected_devices:
            original_bandwidth = self.network_slice["bandwidth"]
            self.network_slice["bandwidth"] = active_bandwidth if device_id in selected_devices else inactive_bandwidth
            effective_bandwidth, latency, packet_loss = self.connect_device(device_id)
            # استدعاء update_network_params لتحديث معلمات الجهاز
            devices[device_id].update_network_params(effective_bandwidth, latency, packet_loss)
            self.network_slice["bandwidth"] = original_bandwidth
        
        logging.info(f"Resources reallocated: Active Bandwidth={active_bandwidth:.2f} Mbps, Inactive Bandwidth={inactive_bandwidth:.2f} Mbps")
        
    def simulate_channel_conditions(self):
        """Simulate dynamic channel conditions (e.g., fading, interference)."""
        self.fading_factor = np.random.uniform(0.9, 1.0) if self.spectrum == "sub6" else np.random.uniform(0.85, 0.95)
        self.interference_factor = np.random.uniform(0.02, 0.05)
        logging.info(f"Channel updated: Fading={self.fading_factor:.2f}, Interference={self.interference_factor:.2f}")

    def get_network_status(self):
        """Return current network status and performance metrics."""
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # دالة خسارة تُستخدم عادةً في مهام التصنيف متعدد الفئات
        self.criterion = nn.CrossEntropyLoss()
        self.local_data = None
        self.effective_bandwidth, self.latency, self.packet_loss = self.fiveg_network.connect_device(self.id)

    # تعديل: إضافة دالة لتحديث effective_bandwidth وlatency وpacket_loss
    def update_network_params(self, effective_bandwidth, latency, packet_loss):
        """Update network parameters for the device."""
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
        # زيادة عداد الاستدعاء للجهاز
        if not hasattr(self, 'call_counter'):
            self.call_counter = 0
        self.call_counter += 1

        if self.local_data is None:
            logging.info(f"Device {self.id}: No data assigned, skipping training.")
            return self.model.state_dict(), 0, 0, 0

        device = torch.device("cpu")
        self.model.to(device)
        self.model.train()

        # حساب نطاق الصور بناءً على العداد
        batch_size = 85
        start_idx = (self.call_counter - 1) * batch_size
        end_idx = start_idx + batch_size

        # التأكد من أن النطاق لا يتجاوز حجم البيانات
        if self.call_counter == 148:
            self.call_counter = 0
            logging.info(f"End the training for device {self.id} and reset the call counter")

        # إنشاء Subset لاختيار 85 صورة بناءً على العداد
        indices = list(range(start_idx, min(end_idx, len(self.local_data))))
        
        # طباعة الـ indices المستخدمة للتحقق من الصور المختارة
        logging.info(f"Device {self.id}, Call {self.call_counter}: Using indices {indices[:10]}{'...' if len(indices) > 10 else ''} (Total: {len(indices)} samples)")

        subset_data = torch.utils.data.Subset(self.local_data, indices)
        
        # طباعة عدد العينات للتحقق من أنها 85 (أو أقل إذا نفدت البيانات)
        logging.info(f"Device {self.id}, Call {self.call_counter}: Training on {len(subset_data)} samples")
    
        loader = torch.utils.data.DataLoader(subset_data, batch_size=batch_size, shuffle=True, drop_last=False)

        # حلقة التدريب
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()


        # Equation 4 from the paper
        T_local = MU_BITS * G / self.cpu_freq if self.cpu_freq > 0 else float('inf')
        # Equation 10 from the paper
        # Energy scale is used
        # يُحسب B_k لتقييم استهلاك الطاقة أثناء تدريب النموذج المحلي على الجهاز، ويُستخدم لتحديد ما إذا كانت الطاقة المتوفرة كافية للتدريب ولتحديث حالة الطاقة
        # لتحويل وحدة الطاقة إلى نطاق مناسب وتجنب مشاكل الدقة العددية، مما يجعل قيمة B_k أسهل في التعامل والتفسير
        B_k = (self.cpu_freq ** 2) * TAU * MU_BITS * G * ENERGY_SCALE
        # Poisson distribution for energy charging
        C_k = self.charge_energy(energy_rate)
        logging.info(f"Device {self.id}: Charging with {C_k} energy units.")
        if self.energy < B_k:
            logging.info(f"Device {self.id}: Insufficient energy ({self.energy:.2f} < {B_k:.2f}), skipping training.")
            return self.model.state_dict(), 0, 0, 0
        # Equation 11 from the paper
        self.energy = max(self.energy - B_k + C_k, 0)
        # Equation 5 from the paper (D = 20 * 1e6)
        T_trans = D / self.effective_bandwidth if self.effective_bandwidth > 0 else float('inf')
        return self.model.state_dict(), T_local, T_trans, B_k

    def report_resources(self):
        return {'cpu_freq': self.cpu_freq, 'energy': self.energy, 'bandwidth': self.effective_bandwidth}

    def __del__(self):
        self.fiveg_network.disconnect_device(self.id)  # تعديل: تصحيح استدعاء disconnect_device

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
        #يُستخدم لتوزيع العينات الزائدة بشكل عادل على الأجهزة الأولى
        remainder = data_size % len(self.devices)
        indices = list(range(data_size))
        random.shuffle(indices)
        start_idx = 0
        for i, device in enumerate(self.devices):
            # Calculate the end index for the current device عشان ما تتكرر البيانات 
            end_idx = start_idx + data_per_device + (1 if i < remainder else 0)
            # Assign a subset of data to the device 
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
            total_bandwidth += device.effective_bandwidth  # Always record bandwidth
            weights, T_local, T_trans, energy_used = device.train_local_model(epochs)
            logging.info(f"Device {device_id}: Energy used: {energy_used} pJ, Bandwidth: {device.effective_bandwidth:.2f} Mbps")
            delay = T_local + T_trans
            delays.append(delay)
            if energy_used > 0:
                local_weights.append(weights)
                #E
                total_energy += energy_used

        if local_weights:
            new_weights = self.fed_avg(local_weights)
            # Update global model with new weights
            # تقوم بتحميل الأوزان أو المعاملات المخزنة في قاموس إلى النموذج العصبي لتحديث حالته
            self.global_model.load_state_dict(new_weights)
        self.energy_history.append(total_energy)
        self.bandwidth_history.append(total_bandwidth)
        #print(f"Round: Total Energy={total_energy}, Total Bandwidth={total_bandwidth:.2f}")
        max_latency = max(delays) if delays else 0
        # تقوم بحساب الحد الأقصى للتأخير (max_latency) من قائمة التأخيرات (delays) التي تم جمعها أثناء التدريب المحلي
        return max_latency, total_energy, total_bandwidth, self.energy_history

    def evaluate_global_model(self, testloader):
        self.global_model.eval()
        correct = 0
        total = 0
        device = torch.device("cpu")
        #  move the model (self.global_model) to a specified device, such as a CPU or GPU
        # تقوم بتقييم نموذج عالمي باستخدام . تضع النموذج في وضع التقييم، تنقله إلى جهاز محدد (مثل CPU)، وتحسب عدد التنبؤات الصحيحة والإجمالية
        self.global_model.to(device)
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                outputs = self.global_model(data)
                # 1: يشير إلى البعد (dimension) الذي نريد أخذ القيمة القصوى منه.
                # _: يتم تجاهل القيم القصوى نفسها، ونأخذ فقط الفئات المتوقعة (predicted).
                _, predicted = torch.max(outputs.data, 1)
                # retrieves the size of the first dimension (dimension 0) of the target tensor,
                # which typically represents the batch size (i.e., the number of samples in the current batch)
                # useful for calculating metrics like accuracy, where you need the total number of samples.
                total += target.size(0)
                # تقوم بحساب عدد التنبؤات الصحيحة من خلال مقارنة القيم المتوقعة (predicted) مع القيم الحقيقية (target).
                # تقوم بجمع عدد التنبؤات الصحيحة في المتغير correct
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
        state = np.array([list(device.report_resources().values()) for device in self.mec_server.devices])
        return state

    def step(self, action):
        selected_indices = [i for i, val in enumerate(action) if val == 1]
        num_selected = len(selected_indices)

        max_latency, total_energy_consumed, total_bandwidth, e = self.mec_server.simulate_training_round(selected_indices)
        #E MAX
        self.e_max_total = np.sum(e)

        # Having a reward in negative means the negative sides of the equation is bigger, 
        # Later on we will choose the biggest reward to select the devices
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
    def __init__(self, input_dim, output_dim):
        self.policy_net = CNN(1, output_dim)
        self.target_net = CNN(1, output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON

    def preprocess_state(self, state):
        state_array = np.array(state).flatten()
        target_size = 256  # 16 × 16
        padded_state = np.pad(state_array, (0, target_size - len(state_array)), mode='constant', constant_values=0)
        return padded_state.reshape(1, 16, 16)

    def select_action(self, state):
        n = len(state)  # عدد الأجهزة الكلي (n)
        action = [0] * n  # تهيئة فضاء الإجراء بـ 0 لكل عميل
        if random.random() < self.epsilon:
            num_selected = random.randint(1, n) # مفروض نكتي الحته دي في الشاتر 
            selected_indices = random.sample(range(n), num_selected)
            for idx in selected_indices:
                action[idx] = 1
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(self.preprocess_state(state), dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1, 16, 16)
                scores = self.policy_net(state_tensor).squeeze()  # Shape: (output_dim,)
                with open("scores_output.txt", "a") as f:
                    f.write(f"Scores: {scores.tolist()}\n")
                num_selected = random.randint(1, n)
                selected_indices = torch.argsort(scores, descending=True)[:num_selected].tolist()
                for idx in selected_indices:
                    action[idx] = 1
        return action

    def optimize_model(self):
        if len(self.memory) < MIN_BATCH_SIZE:
            return
        batch = self.memory.sample(MIN_BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(np.stack([self.preprocess_state(s) for s in states]), dtype=torch.float32)  # Shape: (32, 1, 16, 16)
        next_states = torch.tensor(np.stack([self.preprocess_state(s) for s in next_states]), dtype=torch.float32)  # Shape: (32, 1, 16, 16)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # تم حساب ال  Y  بي قانونها وهي ال predicted
        # تم حساب ال  Q  بالشبكه العصبيه الاونلاين وبعدها تم حساب المتوسط لكل قيم  Q 
        
        # Compute Q-values
        # الشبكة الرئيسية (policy_net) لحساب Q(s, a; θ)
        q_values = self.policy_net(states).squeeze()  # Shape: (batch_size, output_dim)
        # حساب Q(s', a'; θ) لاختيار أفضل a' باستخدام policy_net
        next_q_values_policy = self.policy_net(next_states).detach().squeeze()  # Shape: (batch_size, output_dim)
        # الشبكة الهدف (target_net) لتقييم أفضل a' باستخدام θ' ، Q(s', a'; θ')
        next_q_values_target = self.target_net(next_states).detach().squeeze()  # Shape: (batch_size, output_dim)

        expected_q_values = []
        predicted_q_values = []
        for i in range(MIN_BATCH_SIZE):
            action_indices = [idx for idx, val in enumerate(actions[i]) if val == 1]
            if not action_indices:  # Handle case where no actions are selected
                predicted_q_values.append(torch.tensor(0.0, dtype=torch.float32))
                expected_q_values.append(rewards[i])
                continue
            # 🟦 Q(s, a; θ) = قيم التنبؤ الحالية
            # استخدم action_indices لاختيار Q-values للأفعال المختارة
            q_pred = q_values[i, action_indices]  # Select Q-values for chosen actions
            # 🟦 Q(s, a; θ) = متوسط قيم التنبؤ للأفعال المختارة
            predicted_q_values.append(q_pred.mean())  # Mean Q-value for selected actions
            
            # 🟦 Q(s', a'; θ') = قيم التنبؤ التالية
            q_next = next_q_values_target[i, torch.argmax(next_q_values_policy[i])]  # DDQN update
            # 🟦 Q(s, a; θ) = المكافأة + γ * Q(s', a'; θ')
            expected_q_values.append(rewards[i] + GAMMA * q_next)
        
        predicted_q_values = torch.stack(predicted_q_values)
        expected_q_values = torch.stack(expected_q_values)
        # مابيفرق الترتيب في فنكشن الخساره ، التربيع يزيل الإشارة (سواء كان الفرق موجب أو سالب)
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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=85, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=85, shuffle=False, num_workers=2)

    mec_server = MECServer()
    mec_server.distribute_data(trainset)
    env = DeviceSelectionEnv(mec_server)
    agent = DDQNAgent(INPUT_DIM, NUM_DEVICES)
    rewards_per_episode = []
    episode_energy_history = []
    episode_bandwidth_history = []
    episode_accuracies = []  # قائمة جديدة لتخزين الدقة

    for episode in range(NUM_EPISODES):
        logging.info(f"\nStarting Episode {episode+1}")
        state = env.reset()
        total_reward = 0
        iteration = 0
        avg_reward = float('inf')
        accuracy = 0.0
        episode_energy = 0
        episode_bandwidth = 0
        episode_iterations = 0

        while accuracy < DESIRED_ACCURACY and iteration < MAX_ITERATIONS:
            action = agent.select_action(state)
            next_state, reward = env.step(action)
            agent.memory.push((state, action, reward, next_state))
            agent.optimize_model()
            state = next_state
            total_reward += reward
            iteration += 1
            episode_iterations += 1
            # -1 is used to retrieve the most recent (last) energy value from the energy_history list of the mec_server object.
            # This is likely because the energy_history list stores a sequence of energy values, and the last value represents the most up-to-date or relevant data.
            episode_energy += mec_server.energy_history[-1] if mec_server.energy_history else 0
            episode_bandwidth += mec_server.bandwidth_history[-1] if mec_server.bandwidth_history else 0
            avg_reward = total_reward / iteration if iteration > 0 else reward

            accuracy = mec_server.evaluate_global_model(testloader)
            logging.info(f"Iteration {iteration}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Accuracy: {accuracy:.4f}")
        
        episode_accuracies.append(accuracy)  # تخزين الدقة النهائية للحلقة
        agent.update_epsilon()
        if episode % TARGET_UPDATE == 0:
            # Update the target network every TARGET_UPDATE episodes
            agent.update_target_network()

        rewards_per_episode.append(total_reward)
        episode_energy_history.append(episode_energy / episode_iterations if episode_iterations > 0 else 0)
        episode_bandwidth_history.append(episode_bandwidth / episode_iterations if episode_iterations > 0 else 0)
        logging.info(f"Episode {episode+1} Finished, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, "
                     f"Iterations: {iteration}, Accuracy: {accuracy:.2f}, Energy: {episode_energy_history[-1]:.2f}, "
                     f"Bandwidth: {episode_bandwidth_history[-1]:.2f}")

    # حساب الدقة الكلية
    total_accuracy = np.mean(episode_accuracies) if episode_accuracies else 0
    logging.info(f"Total Accuracy across all episodes: {total_accuracy:.4f}")
    
    # رسم الرسوم البيانية
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
    plt.plot(episode_accuracies, label="Accuracy per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Over Episodes")
    plt.legend()

    plt.tight_layout()
    plt.show()
