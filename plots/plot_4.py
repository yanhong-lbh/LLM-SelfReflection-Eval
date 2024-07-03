import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

fig, ax = plt.subplots()
bar_width = 0.4
opacity = 1.0
hatch_patterns = ['/*/']

index = np.arange(5)

colors = ["#56B4E9", "#E69F00", "#F0E442", "#009E73", "#CC79A7", "#D55E00", "#FFFFFF", "#000000"]

colors_2 = ["#CC3311"] * 5

title_fontsize = 16
label_fontsize = 14

staircase_baseline = [0, 0.25, 0.5, 0.75, 1]
offsets_l = [0.01,0.01,0.01, 0.01, 0]
offsets_r = [-0.01,-0.01,-0.01, -0.01, -0.02]

for i in range(5):
    plt.axhline(y=staircase_baseline[i], xmin=i / 5 + offsets_l[i], xmax=(i + 1) / 5 + offsets_r[i], color=colors_2[i], linestyle='-.', linewidth=3, label='Standard Prompting' if i == 0 else None)
    data = pd.read_csv(f"data/four_diverse_responses_acc_{i}.csv")
    total_e = data[data['level'] == "easy"]["question"].count()
    total_m = data[data['level'] == "medium"]["question"].count()
    total_h = data[data['level'] == "hard"]["question"].count()

    rects1 = plt.bar(index[i] * 3, data[data['level'] == "easy"]["response_with_reflection_acc"].sum() / total_e,
                     bar_width, alpha=opacity, color=colors[2], edgecolor='black', hatch=hatch_patterns[0],
                     label="Self-Reflection (easy)" if i == 0 else None)
    rects2 = plt.bar(index[i] * 3 + bar_width, data[data['level'] == "easy"]["response_without_reflection_acc"].sum() / total_e,
                     bar_width, alpha=opacity, color=colors[2], edgecolor='black', 
                     label="Exploration-Only (easy)" if i == 0 else None)
    rects3 = plt.bar(index[i] * 3 + 2 * bar_width, data[data['level'] == "medium"]["response_with_reflection_acc"].sum() / total_m,
                     bar_width, alpha=opacity, color=colors[0], edgecolor='black', hatch=hatch_patterns[0], 
                     label="Self-Reflection (medium)" if i == 0 else None)
    rects4 = plt.bar(index[i] * 3 + 3 * bar_width, data[data['level'] == "medium"]["response_without_reflection_acc"].sum() / total_m,
                     bar_width, alpha=opacity, color=colors[0], edgecolor='black',
                     label="Exploration-Only (medium)" if i == 0 else None)
    rects5 = plt.bar(index[i] * 3 + 4 * bar_width, data[data['level'] == "hard"]["response_with_reflection_acc"].sum() / total_h,
                     bar_width, alpha=opacity, color=colors[1], edgecolor='black', hatch=hatch_patterns[0], 
                     label="Self-Reflection (hard)" if i == 0 else None)
    rects6 = plt.bar(index[i] * 3 + 5 * bar_width, data[data['level'] == "hard"]["response_without_reflection_acc"].sum() / total_h,
                     bar_width, alpha=opacity, color=colors[1], edgecolor='black',
                     label="Exploration-Only (hard)" if i == 0 else None)
    

plt.xlabel("Response Accuracy", fontsize=label_fontsize)
plt.title("Accuracy Decomposition over 4 Responses", fontsize=title_fontsize)

ax.set_xticks(index * 3 + 1)
x_labels = ['0%', '25%', '50%', '75%', '100%']
ax.set_xticklabels([x_labels[i] for i in range(5)], fontsize=label_fontsize)
ax.tick_params(axis='y', labelsize=label_fontsize)


with PdfPages('normalized_average_acc_by_total_acc_percentage_level_4.pdf') as pdf:
    pdf.savefig(fig, bbox_inches='tight')
plt.show()

