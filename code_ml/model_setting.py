import numpy as np
import pandas as pd
from regressionML import RegressionML
from classificationML import ClassificationML
from sklearn.preprocessing import MinMaxScaler


class Model_Setting():
    def __init__(self, regression=True):
        self.cf = {}
        self.cf['file'] = '../data/20200203_UCF_Env_Data_DD_updated_for_paper_7-31-2020.xlsx'
        self.cf['sheet'] = 'Image analysis'
        self.cf['targets'] = ['OV', 'Oil_separation(%)', 'Turbidity']
        # self.cf['targets'] = ['Turbidity']
        self.cf['target'] = 'OV'
        self.cf['selected_Xs'] = ['Surfactant name',
                                  'Surfactant concentration (ppm)',
                                  'Critical micelle concentration (CMClog) (ppm)',
                                  'Equilibrium surface tension (ST) above CMC (air) (mN/M)',
                                  'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)',
                                  'Micelle size (nm)',
                                  'Zeta potential (mV)',
                                  'pH',
                                  'Suspended solids concentration (ppm)',
                                  'Salinity (ppm)',
                                  'Temperature (°C)',
                                  'Alkalinity (mg CaCO3/L)',
                                  "Surfactant's Initial pH at 7CMC"]

        self.cf['regression'] = regression # False means 'classification'
        self.cf['normalization'] = [False]  # False means 'non-normalization'

        if self.cf['regression'] is True:
            self.ml = RegressionML()
        else:
            self.ml = ClassificationML()

    def convert_to_categorical_variable(self, df_y, normalization):
        if self.cf['regression'] is True:
            return df_y

        # Below is for classification
        if normalization is False:
            df_y = MinMaxScaler(feature_range=(0, 1)).fit_transform(df_y.to_frame())
            df_y = pd.Series(df_y[:, 0])

        bins = [np.NINF, 0.33, 0.66, np.inf]
        names = ['low', 'mid', 'high']
        df_y = pd.cut(df_y, bins, labels=names)

        return np.ravel(df_y)
