# Plot code generated with the help of ChatGPT

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D

# Iterations
iterations = np.array([1, 2, 3, 4, 5])

# Metrics for each model
first_order = {
    'Hull Volume': [0.046770738, 0.026793184, 0.029602272, 0.024692083, 0.022256572],
    'MRAE': [1.0286954, 1.025031, 1.0304126, 1.0306946, 1.03175425],
    'RMSE': [1.4219018, 1.4204112, 1.4253812, 1.4249156, 1.426095],
    'R2': [0.8990572, 0.9128818, 0.9133478, 0.9138, 0.9123152]
}

second_order = {
    'Hull Volume': [0.000105496, 3.99152E-05, 0.000027779, 0.00002274, 1.14826E-05],
    'MRAE': [1.0048792, 1.0117278, 1.0257752, 1.0509664, 1.035074],
    'RMSE': [1.3178704, 1.369947, 1.4116364, 1.4240176, 1.4159918],
    'R2': [0.8446546, 0.8489114, 0.8428054, 0.8355234, 0.8416042]
}

water_tank = {
    'Hull Volume': [0.021414141, 0.008556678, 0.003250281, 0.002065095, 0.001104377],
    'MRAE': [0.8229405, 0.7980438, 0.7807624, 0.7836012, 0.775818],
    'RMSE': [0.77152525, 0.7566254, 0.751232, 0.7492782, 0.7477818],
    'R2': [0.910146, 0.9142104, 0.9134558, 0.913672, 0.9135308]
}

# Standard deviations for error bars
std_devs = {
    'First Order': {
        'Hull Volume': [0, 0.00723676, 0.006732827, 0.003798252, 0.003417126],
        'MRAE': [0, 0.001289435, 0.001720281, 0.002457956, 0.000215437],
        'RMSE': [0, 0.000343992, 0.001080604, 0.002237646, 0.000316786],
        'R2': [0, 0.005925712, 0.005713853, 0.000263339, 0.000631176]
    },
    'Second Order': {
        'Hull Volume': [0, 2.66009E-05, 8.0252E-06, 2.58239E-05, 4.42893E-06],
        'MRAE': [4.04113E-05, 0.017744402, 0.031936213, 0.030726213, 0.036344323],
        'RMSE': [0.000346217, 0.02549405, 0.03685602, 0.02899213, 0.03495616],
        'R2': [0.00018915, 0.006401916, 0.010658642, 0.010228113, 0.012547321]
    },
    'Water Tank': {
        'Hull Volume': [0, 0.006682858, 0.002577412, 0.001346203, 0.00084265],
        'MRAE': [0.002482108, 0.021221929, 0.016390398, 0.004775869, 0.013627047],
        'RMSE': [0.001627089, 0.012635557, 0.006381661, 0.005003515, 0.005150452],
        'R2': [5.69446E-05, 0.001751892, 0.001615007, 0.00030731, 0.001157541]
    }
}


models = {'First Order': first_order, 'Second Order': second_order, 'Water Tank': water_tank}
metrics = ['Hull Volume', 'MRAE', 'RMSE', 'R2']
colors = {'First Order': 'green', 'Second Order': 'red', 'Water Tank': 'blue'}
markers = {'First Order': 'o', 'Second Order': 's', 'Water Tank': '^'}

fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs = axs.ravel()

for i, metric in enumerate(metrics):
    ax = axs[i]
    for model_name, model_data in models.items():
        y = np.array(model_data[metric])
        ax.scatter(iterations, y, label=model_name, color=colors[model_name], marker=markers[model_name])

        # Add error bars
        yerr = np.array(std_devs[model_name][metric])
        ax.errorbar(iterations, y, yerr=yerr, fmt=markers[model_name], label=model_name,
                    color=colors[model_name], capsize=4, linestyle='None')
        # Trendline

        # Fit and plot trendline
        weights = 1 / (yerr ** 2 + 1e-8)  # Add epsilon to avoid division by zero
        z = np.polyfit(iterations, y, 2, w=weights)
        p = np.poly1d(z)
        # ax.plot(iterations, p(iterations), linestyle='--', color=colors[model_name])
        ax.plot(iterations, y, linestyle='--', color=colors[model_name])


    ax.set_title(metric, fontsize=16)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.set_xticks([1,2,3,4,5])

    if i == 0:  # First subplot (Hull Volume)
        ax.set_yscale('log')
        ax.set_ylabel("Log Hull Volume")


# Create custom legend handles (just the dataset markers)
legend_handles = [
    Line2D([0], [0], marker=markers[name], color='w', markerfacecolor=colors[name],
           label=name, markersize=10, linestyle='None')
    for name in models.keys()
]

fig.suptitle("Model Performance over SPS Iterations", fontsize=18)
fig.tight_layout(rect=[0, 0.1, 1, 1])
fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=14)
plt.show()
