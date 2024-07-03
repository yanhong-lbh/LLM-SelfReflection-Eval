import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(11, 4))
ax.set_ylim(bottom=0.2)
bar_width = 0.4
opacity = 1
hatch_patterns = ['/*']

index = np.arange(11)

colors = ["#56B4E9", "#E69F00", "#F0E442", "#009E73", "#CC79A7", "#D55E00"]
colors_baseline = ["#CC3311"] * 11
name = ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0']
x_labels = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']

title_fontsize = 20
label_fontsize = 18

staircase_baseline = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
offsets_l = [0,0,0,0.02,0.01,0, 0,-0.01,-0.02,-0.03,-0.03]
offsets_r = [-0.03,-0.03,-0.03, 0.01, -0,-0,-0.01, -0.01,-0.02, -0.03,-0.03]

data = pd.read_csv(f"data/artificial_responses_10.csv")

for i in range(11):
    total_e = data[data['level'] == "easy"]["question"].count()
    total_m = data[data['level'] == "medium"]["question"].count()
    total_h = data[data['level'] == "hard"]["question"].count()
    n = name[i]

    rects1 = plt.bar(index[i] * 3, data[data['level'] == "easy"][f"res_w_ref_i{n}_acc"].sum() / total_e,
                     bar_width, alpha=opacity, color=colors[2], edgecolor='black', hatch=hatch_patterns[0], 
                     label="Self-Reflection (easy)" if i == 0 else None)
    rects2 = plt.bar(index[i] * 3 + bar_width, data[data['level'] == "easy"][f"res_wo_ref_i{n}_acc"].sum() / total_e,
                     bar_width, alpha=opacity, facecolor=colors[2], edgecolor='black',
                     label="Exploration-Only (easy)" if i == 0 else None)
    rects3 = plt.bar(index[i] * 3 + 2 * bar_width, data[data['level'] == "medium"][f"res_w_ref_i{n}_acc"].sum() / total_m,
                     bar_width, alpha=opacity, color=colors[0], edgecolor='black', hatch=hatch_patterns[0], 
                     label="Self-Reflection (medium)" if i == 0 else None)
    rects4 = plt.bar(index[i] * 3 + 3 * bar_width, data[data['level'] == "medium"][f"res_wo_ref_i{n}_acc"].sum() / total_m,
                     bar_width, alpha=opacity, color=colors[0], edgecolor='black',
                     label="Exploration-Only (medium)" if i == 0 else None)
    rects5 = plt.bar(index[i] * 3 + 4 * bar_width, data[data['level'] == "hard"][f"res_w_ref_i{n}_acc"].sum() / total_h,
                     bar_width, alpha=opacity, color=colors[1], edgecolor='black', hatch=hatch_patterns[0], 
                     label="Self-Reflection (hard)" if i == 0 else None)
    rects6 = plt.bar(index[i] * 3 + 5 * bar_width, data[data['level'] == "hard"][f"res_wo_ref_i{n}_acc"].sum() / total_h,
                     bar_width, alpha=opacity, color=colors[1], edgecolor='black',
                     label="Exploration-Only (hard)" if i == 0 else None)

    plt.axhline(y=staircase_baseline[i], xmin=i / 11 + offsets_l[i], xmax=(i + 1) / 11 + offsets_r[i], color=colors_baseline[i], linestyle='-.', linewidth=3, label='Standard Prompting' if i == 0 else None)

plt.xlabel("Response Accuracy", fontsize=label_fontsize)
plt.title("Accuracy Decomposition with Artificial Responses (K=10)", fontsize=title_fontsize)

ax.set_xticks(index * 3 + 1)
ax.set_xticklabels([x_labels[i] for i in range(11)], fontsize=label_fontsize)
ax.tick_params(axis='y', labelsize=label_fontsize)

linewidth_value = 2

plt.axvline(x=5.5, linestyle='dotted', color='black', linewidth=linewidth_value)
plt.annotate("Easy", xy=(0, 0.8),
             xytext=(0.21, 0.8), 
             xycoords='data', textcoords='axes fraction', fontsize=label_fontsize)

plt.axvline(x=8.5, linestyle='dotted', color='black', linewidth=linewidth_value)
plt.annotate("Medium", xy=((index[3] * 3 + 1) - 1.5 * bar_width, 0.8),
             xytext=(0.295, 0.8), 
             xycoords='data', textcoords='axes fraction', fontsize=label_fontsize)

plt.axvline(x=14.5, linestyle='dotted', color='black', linewidth=linewidth_value)
plt.annotate("Hard", xy=((index[5] * 3 + 1) - 1.5 * bar_width, 0.8),
             xytext=(0.462, 0.8), 
             xycoords='data', textcoords='axes fraction', fontsize=label_fontsize)

with PdfPages('normalized_average_acc_by_total_acc_percentage_level_10.pdf') as pdf:
    pdf.savefig(fig, bbox_inches='tight')
plt.show()