import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

o_df = pd.read_excel(open('../output_for_paper/[9-3-2020] Reg_ML_OV_OS.xlsx', 'rb'), sheet_name='OS', index_col=None, header=None)
# o_df = pd.read_excel(open('../output_for_paper/[9-3-2020] Reg_ML_OV_OS.xlsx', 'rb'), sheet_name='OV', index_col=None, header=None)

# Columns
# Actual | GB | RF

# Show data points
x = np.arange(o_df.shape[0])
marker = ["*", "4", "^", "D", "+", "_", ".", "3", "x"]
scatters = []

scatters.append(plt.scatter(o_df[0], o_df[1].to_numpy(), marker=".", alpha=0.5, c='b'))
scatters.append(plt.scatter(o_df[0], o_df[2].to_numpy(), marker="x", alpha=0.5, c='k'))

# Show X-Axis
# df = pd.DataFrame({"id":['SFC', 'CMC', 'EST', 'EIT', 'MCS', 'ZTP', 'AKL', "SIP", 'PH', 'SSC', 'SLT', 'TPR']})
ax = plt.gca()
# ax.xaxis.set_ticks(np.arange(len(df['id'])))
# ax.xaxis.set_ticklabels(df['id'], rotation=90)

ax.set_xlim([-.1, 1.1])
ax.set_ylim([-.1, 1.1])
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# Show Line
# ax.plot([0, o_df.shape[0]], [0, 0], '--', linewidth=0.5, c='k', alpha=0.5)

# Show Legend
plt.legend(scatters, ["GB", "RF"])

plt.show()