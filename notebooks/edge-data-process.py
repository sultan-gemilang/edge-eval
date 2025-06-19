# --- Parse tegrastats log and plot metrics ---

import re
import pandas as pd
import matplotlib.pyplot as plt

# Path to your log file
logfile = "logs/edge/t5-small/t5-small_20250619_015845_tegrastats.log"

# Prepare lists for parsed data
timestamps, ram_used, cpu_avg, gr3d_freq, cpu_temp, gpu_temp, power = [], [], [], [], [], [], []

with open(logfile) as f:
    for line in f:
        # Timestamp
        ts = line[:19]
        timestamps.append(ts)
        # RAM used
        m = re.search(r'RAM (\d+)/', line)
        ram_used.append(int(m.group(1)) if m else None)
        # CPU average usage
        m = re.findall(r'CPU \[([^\]]+)\]', line)
        if m:
            cpu_vals = [int(x.split('%')[0]) for x in m[0].split(',')]
            cpu_avg.append(sum(cpu_vals)/len(cpu_vals))
        else:
            cpu_avg.append(None)
        # GPU freq
        m = re.search(r'GR3D_FREQ (\d+)%', line)
        gr3d_freq.append(int(m.group(1)) if m else None)
        # CPU temp
        m = re.search(r'cpu@([\d\.]+)C', line)
        cpu_temp.append(float(m.group(1)) if m else None)
        # GPU temp
        m = re.search(r'gpu@([\d\.]+)C', line)
        gpu_temp.append(float(m.group(1)) if m else None)
        # Power
        m = re.search(r'VDD_IN (\d+)mW', line)
        power.append(int(m.group(1)) if m else None)

# Build DataFrame
df = pd.DataFrame({
    "Time": pd.to_datetime(timestamps),
    "RAM_Used_MB": ram_used,
    "CPU_Usage_%": cpu_avg,
    "GPU_Usage_%": gr3d_freq,
    "CPU_Temp_C": cpu_temp,
    "GPU_Temp_C": gpu_temp,
    "Power_mW": power
})

# Plot
fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
df.plot(x="Time", y="RAM_Used_MB", ax=axs[0,0], title="RAM Used (MB)")
df.plot(x="Time", y="CPU_Usage_%", ax=axs[0,1], title="CPU Usage (%)")
df.plot(x="Time", y="GPU_Usage_%", ax=axs[1,0], title="GPU Usage (%)")
df.plot(x="Time", y="CPU_Temp_C", ax=axs[1,1], title="CPU Temp (C)")
df.plot(x="Time", y="GPU_Temp_C", ax=axs[2,0], title="GPU Temp (C)")
df.plot(x="Time", y="Power_mW", ax=axs[2,1], title="Power (mW)")
plt.tight_layout()
plt.show()