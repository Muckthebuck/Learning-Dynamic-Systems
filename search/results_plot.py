import matplotlib.pyplot as plt

# Dataset
epochs = [1, 2, 3, 4, 5, 6]
hull_volume = [0.025834, 0.015672, 0.012182, 0.012182, 0.010162, 0.010162]
mrae = [1.128817, 1.084777, 1.1184, 1.116554, 1.112808, 1.119155]
rmse = [1.464263, 1.411854, 1.455037, 1.447802, 1.45097, 1.457235]
r2 = [0.827178, 0.843288, 0.846544, 0.849494, 0.842298, 0.838389]

# Create 2x2 subplots with a main title
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('ARMAX First Order', fontsize=16)

# Plot each metric
axs[0, 0].scatter(epochs, hull_volume, color='blue')
axs[0, 0].set_title('Hull Volume')
axs[0, 0].set_xlabel('Iteration')
axs[0, 0].set_ylabel('Hull Volume')

axs[0, 1].scatter(epochs, mrae, color='green')
axs[0, 1].set_title('MRAE')
axs[0, 1].set_xlabel('Iteration')
axs[0, 1].set_ylabel('MRAE')

axs[1, 0].scatter(epochs, rmse, color='red')
axs[1, 0].set_title('RMSE')
axs[1, 0].set_xlabel('Iteration')
axs[1, 0].set_ylabel('RMSE')

axs[1, 1].scatter(epochs, r2, color='purple')
axs[1, 1].set_title('R2')
axs[1, 1].set_xlabel('Iteration')
axs[1, 1].set_ylabel('R2')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()