from saveResults import SaveResults
from datetime import datetime


class Experiment():
    def __init__(self):

        # Set configuration
        self.cf = {}
        self.cf['file'] = '../data/20200203_UCF_Env_Data_DD_updated_for_paper_9-1-2020.xlsx'
        self.cf['sheet'] = 'Image analysis'

        self.cf['target'] = None
        self.cf['selected_Xs'] = ['Surfactant concentration (ppm)',
                                  'Critical micelle concentration (CMClog) (ppm)',
                                  'Equilibrium surface tension (ST) above CMC (air) (mN/M)',
                                  'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)',
                                  'Micelle size (nm)',
                                  'Zeta potential (mV)',
                                  'initial pH',
                                  'Suspended solids concentration (ppm)',
                                  'Salinity (ppm)',
                                  'Temperature (°C)',
                                  'Alkalinity (mg CaCO3/L)',
                                  "Surfactant's Initial pH at 7CMC"
                                ]

        # self.cf['n_splits'] = 0
        self.cf['n_splits'] = 10

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
