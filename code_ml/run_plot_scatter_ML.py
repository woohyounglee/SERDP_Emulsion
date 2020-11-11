import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

# sheet_name = 'OS'
sheet_name = 'OV'
o_df = pd.read_excel(open('../output_for_paper/[11-10-2020] Reg_ML_OV_OS.xlsx', 'rb'), sheet_name=sheet_name, index_col=None, header=None)

# Columns
# Actual | DT or GB | RF
if sheet_name == 'OS':
    x = np.arange(o_df.shape[0])
    scatters = []
    scatters.append(plt.scatter(o_df[0], o_df[1].to_numpy(), marker=".", alpha=0.5, c='b'))
    scatters.append(plt.scatter(o_df[0], o_df[2].to_numpy(), marker="x", alpha=0.5, c='k'))
    ax = plt.gca()
    ax.set_xlim([-.1, 1.1])
    ax.set_ylim([-.1, 1.1])
    plt.legend(scatters, ["GB", "RF"])
    plt.show()
elif sheet_name == 'OV':
    x = np.arange(o_df.shape[0])
    scatters = []
    scatters.append(plt.scatter(o_df[0], o_df[1].to_numpy(), marker=".", alpha=0.5, c='b'))
    scatters.append(plt.scatter(o_df[0], o_df[2].to_numpy(), marker="x", alpha=0.5, c='k'))
    ax = plt.gca()
    ax.set_xlim([-.1, 1.1])
    ax.set_ylim([-.1, 1.1])
    plt.legend(scatters, ["DT", "RF"])
    plt.show()