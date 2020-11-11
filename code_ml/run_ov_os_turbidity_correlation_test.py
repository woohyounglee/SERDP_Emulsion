from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from matplotlib.pyplot import figure


class Experiment():
    def __init__(self):
        self.cf = {}
        self.cf['file'] = '../data/20200203_UCF_Env_Data_DD_updated_for_paper_11-10-2020.xlsx'
        self.cf['sheet'] = 'Image analysis'
        self.cf['target'] = ['OV (Oily value)', 'Turbidity']
        # self.cf['target'] = ['OS (Oil separation)', 'Turbidity']
        # self.cf['target'] = ['Oil_separation(%)', 'OV']

        excel_data = pd.read_excel(self.cf['file'], self.cf['sheet'])

        #  Select variables
        data = excel_data[self.cf['target']]

        # Remove data if it contains 'nan'
        self.data = data.dropna()

        # Normalize data
        scaled_df = MinMaxScaler(feature_range=(0, 1)).fit_transform(self.data)
        self.data = pd.DataFrame(scaled_df, index=self.data.index, columns=self.data.columns)

    def run(self):
        x = self.data[self.cf['target'][0]]
        y = self.data[self.cf['target'][1]]

        pearsonr =["{:.15f}".format(float(i)) for i in scipy.stats.pearsonr(x, y)]
        kendalltau = ["{:.15f}".format(float(i)) for i in scipy.stats.kendalltau(x, y)]
        spearmanr = ["{:.15f}".format(float(i)) for i in scipy.stats.spearmanr(x, y)]

        print('pearsonr:', pearsonr)
        print('kendalltau:', kendalltau)
        print('spearmanr:', spearmanr)

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r_value={r_value:.2f}, p_value={p_value:.2f}, std_err={std_err:.2f}'

        print(line)

        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)

        ax.plot(x, y, linewidth=0, marker='s', label='Data points')
        ax.plot(x, intercept + slope * x)
        # ax.plot(x, intercept + slope * x, label=line)
        ax.set_xlabel(self.cf['target'][0])
        ax.set_ylabel(self.cf['target'][1])
        ax.legend(facecolor='white')
        plt.show()

Experiment().run()
