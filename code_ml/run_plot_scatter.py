import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

# os_df = pd.read_excel(open('../output_for_paper/[9-4-2020] Removal Test.xlsx', 'rb'), sheet_name='OS', index_col=None, header=None)
os_df = pd.read_excel(open('../output_for_paper/[9-4-2020] Removal Test.xlsx', 'rb'), sheet_name='OV', index_col=None, header=None)


# Show data points
x = np.arange(os_df.shape[0])
marker = ["*", "4", "^", "D", "+", "_", ".", "3", "x"]
scatters = []
for i in range(os_df.shape[1]):
    scatters.append(plt.scatter(x, os_df[i].to_numpy(), marker=marker[i], alpha=0.5, c='k'))

# Show X-Axis
df = pd.DataFrame({"id":['SFC', 'CMC', 'EST', 'EIT', 'MCS', 'ZTP', 'AKL', "SIP", 'PH', 'SSC', 'SLT', 'TPR']})
ax = plt.gca()
ax.xaxis.set_ticks(np.arange(len(df['id'])))
ax.xaxis.set_ticklabels(df['id'], rotation=90)

ax.set_ylim([-0.04, 0.13])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# Show Line
ax.plot([0, os_df.shape[0]], [0, 0], '--', linewidth=0.5, c='k', alpha=0.5)

# Show Legend
plt.legend(scatters, ["DL", "BR", "RF", "DT", "GB", "MP", "LR", "GP", "SV"])

plt.show()