import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Js = np.load(snakemake.input["Js"])
print(Js)

trajectories = pd.read_csv(snakemake.input["results"])

r = np.array(trajectories.reward)
r = np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid')


plt.subplot(2, 1, 1)
plt.plot(Js)
plt.subplot(2, 1, 2)
plt.plot(r)
plt.show()
#legend_labels.append(names[a])
