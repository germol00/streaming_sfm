import pandas as pd
import matplotlib.pyplot as plt
import re

def parse_beam_file(filename, policy_name):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # Handle potential prefix
            clean_line = re.sub(r'\\s*', '', line)
            parts = clean_line.split()
            if len(parts) == 4:
                try:
                    chunk_str, wer, aware, unaware = parts
                    chunk_size = float(chunk_str.replace('_', '.'))
                    data.append({
                        'chunk_size': chunk_size,
                        'WER': float(wer),
                        'Aware_Latency': float(aware),
                        'Unaware_Latency': float(unaware),
                        'Policy': policy_name
                    })
                except ValueError: continue
    return pd.DataFrame(data)

# 1. Load your Beam 32 data files
files_beam32 = {
    'SLCP': 'res_beam_32_pol_SLCP.txt',
    'LACP': 'res_beam_32_pol_LACP.txt',
    'LCP': 'res_beam_32_pol_LCP.txt'
}

all_data = []
for policy, fname in files_beam32.items():
    all_data.append(parse_beam_file(fname, policy))

df = pd.concat(all_data).sort_values(['Policy', 'chunk_size'])

# 2. Setup the dual-axis plot
fig, ax1 = plt.subplots(figsize=(9, 9))
ax2 = ax1.twinx()

ax1.set_box_aspect(1) 

colors = {'LCP': 'tab:red', 'LACP': 'tab:green', 'SLCP': 'tab:blue'}
markers = {'LCP': 'o', 'LACP': 's', 'SLCP': '^'}

# Lists to collect legend entries in the grouped order
legend_handles = []
legend_labels = []

# 3. Plotting loop
for policy in ['LCP', 'LACP', 'SLCP']:
    subset = df[df['Policy'] == policy]
    if subset.empty: continue

    # Plot Aware Latency (Left Axis)
    h_aware, = ax1.plot(subset['chunk_size'], subset['Aware_Latency'],
                        color=colors[policy], linestyle='-', marker=markers[policy])

    # Plot Unaware Latency (Left Axis)
    h_unaware, = ax1.plot(subset['chunk_size'], subset['Unaware_Latency'],
                          color=colors[policy], linestyle=':', marker=markers[policy])

    # Plot WER (Right Axis)
    h_wer, = ax2.plot(subset['chunk_size'], subset['WER'],
                      color=colors[policy], linestyle='--', linewidth=2.5, marker='x')

    # Explicitly group items for this policy in the legend list
    legend_handles.extend([h_aware, h_unaware, h_wer])
    legend_labels.extend([f'{policy} Aware Latency', f'{policy} Unaware Latency', f'{policy} WER'])

# 4. Final Formatting
#ax1.set_xlabel('Chunk Size (s)', fontsize=12)
#ax1.set_ylabel('Latency (s)', fontsize=12)
#ax2.set_ylabel('WER (%)', fontsize=12)
ax1.text(0.02, 0.97, 'Latency (s)', transform=ax1.transAxes,
             fontweight='bold', fontsize=12)
ax1.text(0.88, 0.97, 'WER (%)', transform=ax1.transAxes,
             fontweight='bold', fontsize=12)
ax1.text(0.02, 0.02, r'$\mathbf{L_c}$ (s)', transform=ax1.transAxes,
             fontweight='bold', fontsize=12)
ax1.set_ylim(0.6, 3.90)
ax2.set_ylim(6.63, 8.49)
ax1.grid(True, linestyle='--', alpha=0.5)

# Apply the grouped legend
ax1.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(0.18, 0.98), fontsize='small')

plt.tight_layout()
#plt.show()
plt.savefig('asr_policy_latency.pdf')
