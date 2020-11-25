import numpy as np
import pandas as pd
from regressionML import RegressionML
from classificationML import ClassificationML
from sklearn.preprocessing import MinMaxScaler
from saveResults import SaveResults
from datetime import datetime


class Model_Setting():
    def __init__(self, regression=True):
        self.cf = {}
        self.cf['file'] = '../data/20200203_UCF_Env_Data_DD_updated_for_paper_11-10-2020.xlsx'
        self.cf['sheet'] = 'Image analysis'
        self.cf['targets'] = ['OV (Oily value)', 'OS (Oil separation)', 'Turbidity']
        # self.cf['targets'] = ['Turbidity']
        self.cf['target'] = 'OV (Oily value)'
        # self.cf['target'] = None
        self.cf['selected_Xs'] = ['Critical micelle concentration (CMClog) (ppm)',
                                  'Equilibrium surface tension (ST) above CMC (air) (mN/M)',
                                  'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)',
                                  'Micelle size (nm)',
                                  'Zeta potential (mV)',
                                  'Alkalinity (mg CaCO3/L)',
                                  "Surfactant's Initial pH at 7CMC",
                                  'Surfactant concentration (ppm)',
                                  'pH',
                                  'Suspended solids concentration (ppm)',
                                  'Salinity (ppm)',
                                  'Temperature (°C)', ]

        self.cf['regression'] = regression # False means 'classification'
        self.cf['normalization'] = [False]  # False means 'non-normalization'

        if self.cf['regression'] is True:
            self.ml = RegressionML()
        else:
            self.ml = ClassificationML()

        # ========================================================
        # Define the number of data split
        self.cf['n_splits'] = 10

        # Define excel data for storing all data to Excel
        self.init_excel_file()

    def init_excel_file(self):
        # Define excel data for storing all data to Excel
        self.excel = {}
        self.excel['X Variables'] = []
        self.excel['Target'] = []
        self.excel['Regression'] = []

        self.excel['Train Size'] = []
        self.excel['Test Size'] = []
        self.excel['X Removed'] = []
        self.excel['F Results'] = []

        # set model info
        for v in self.cf['selected_Xs']:
            self.excel['X Variables'].append(v)

    def save_excel_file(self):
        experiment_time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        excel_file = f'../output/{experiment_time}.xlsx'
        excel_experiment = SaveResults(excel_file)
        for k, l in self.excel.items():
            excel_experiment.insert(k, l)
        excel_experiment.save()

    def convert_to_categorical_variable(self, df_y):
        if self.cf['regression'] is True:
            return df_y

        bins = [np.NINF, 0.33, 0.66, np.inf]
        names = ['low', 'mid', 'high']
        df_y = pd.cut(df_y, bins, labels=names)

        return np.ravel(df_y)
