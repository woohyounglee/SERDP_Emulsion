import numpy as np
import pandas as pd
from regressionML import RegressionML
from classificationML import ClassificationML
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from saveResults import SaveResults
from datetime import datetime
from sklearn.model_selection import KFold
from statistics import stdev
from statistics import mean
import winsound
from sklearn.feature_selection import f_regression, mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns


class Experiment():
    def __init__(self):
        # self.size_experiments = 2
        self.size_experiments = 10

        # Set configuration
        self.cf = {}
        self.cf['file'] = '../data_second_experiment/Emulaion_data_9-9-2021.xlsx'
        self.cf['targets'] = ['OS (%)', 'Turbidity']
        self.cf['target'] = None

        self.cf['sheet'] = 'Dataset(1)'
        # self.cf['sheet'] = 'Dataset(2)'
        # self.cf['sheet'] = 'Dataset(3)'

        self.cf['selected_Xs'] = ['Surfactant name',
                                  'CMC (ppm)',
                                  'Equilibrium ST (mN/M)',
                                  'Equilibrium IFT (mN/M)',
                                  'Micelle size (nm)',
                                  'Zeta potential (mV)',
                                  'Alkalinity (mg CaCO3/L)',
                                  "Surfactant's pH",
                                  'pH',
                                  'SS (ppm)',
                                  'Salinity (ppm)',
                                  'CMC',
                                  'Temperature (°C)',
                                  'Mix']


        # self.cf['regression'] = True  # False means 'classification'
        self.cf['regression'] = False

        # self.cf['all_surfactants'] = [True, False] # False means the each-surfactant experiment
        self.cf['all_surfactants'] = [True]
        # self.cf['normalization'] = [True, False] # False means 'non-normalization
        self.cf['normalization'] = [True]
        self.cf['sensitivity analysis'] = False
        # self.cf['sensitivity analysis'] = True

        # Define excel data for storing all data to Excel
        self.excel = {}
        self.excel['X Variables'] = []
        self.excel['Target'] = []
        self.excel['Regression'] = []
        self.excel["Surfactants"] = []
        self.excel['Normalization'] = []
        self.excel['Train Size'] = []
        self.excel['Test Size'] = []
        self.excel['X Removed'] = []
        self.excel['F Results'] = []

        if self.cf['regression'] is True:
            self.ml = RegressionML()
            for clf in self.ml.regressors:
                name = type(clf).__name__
                self.excel[name + '-MAE'] = []
            for clf in self.ml.regressors:
                name = type(clf).__name__
                self.excel[name + '-MAE-STD'] = []
        else:
            self.ml = ClassificationML()
            for clf in self.ml.classifiers:
                name = type(clf).__name__
                self.excel[name + '-ACC'] = []
            for clf in self.ml.classifiers:
                name = type(clf).__name__
                self.excel[name + '-ACC-STD'] = []

        # set model info
        for v in self.cf['selected_Xs']:
            self.excel['X Variables'].append(v)

    def convert_to_categorical_variable(self, df_y, normalization):
        if self.cf['regression'] is True:
            return df_y

        # Below is for classification
        if normalization is False:
            df_y = MinMaxScaler(feature_range=(0, 1)).fit_transform(df_y.to_frame())
            df_y = pd.Series(df_y[:,0])

        # trisection: 3 equal parts
        bins = [np.NINF, 0.33, 0.66, np.inf]
        names = ['low', 'mid', 'high']

        # bisection: 2 equal parts
        # bins = [np.NINF, 0.5, np.inf]
        # names = ['low', 'high']

        df_y = pd.cut(df_y, bins, labels=names)

        return np.ravel(df_y)

    def convert_to_discrete_variable(self, data, column='Mix'):
        enc = OrdinalEncoder()
        enc.fit(data[[column]])
        data[[column]] = enc.transform(data[[column]])



    def save_excel_file(self):
        experiment_time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        excel_file = f'../output/{self.cf["sheet"]}_regression[{self.cf["regression"]}]_{experiment_time}.xlsx'
        excel_experiment = SaveResults(excel_file)
        for k, l in self.excel.items():
            excel_experiment.insert(k, l)
        excel_experiment.save()

    def show_heatmap_matrix(self, df, columns):
        cm = np.corrcoef(df[columns].values.T)
        sns.set(font_scale=0.8)
        # hm = sns.heatmap(cm, char=True, annot=True, square=True, fmt='.2f', annot_kws={'size':15}, yticklabels=columns, xticklabels=columns)
        hm = sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=columns,
                         xticklabels=columns)
        plt.show()
        plt.close()

    def get_f_regression(self, X_train, y_train):
        f_test, _ = f_regression(X_train, y_train)

        results_f_dict = {}
        for i in range(X_train.shape[1]):
            results_f_dict[X_train.columns.values[i]] = f_test[i]

        results_f_dict = {k: v for k, v in sorted(results_f_dict.items(), key=lambda item: item[1], reverse=True)}

        print("======================")
        print("=       F-test       =")
        f_results = ''
        for k, v in results_f_dict.items():
            f_results += f'{k}\t{v}\r'
            print(f'{k}\t{v}')

        return f_results

    def run_ml_removing_sensitivity_analysis(self, surfactant='All'):
        # 1. Load data from xlsx
        excel_data = pd.read_excel(self.cf['file'], self.cf['sheet'])

        for X in self.cf['selected_Xs']:
            selected_Xs = self.cf['selected_Xs'].copy()
            selected_Xs.remove(X)
            print(f'******************** removed {X} ********************')
            for target in self.cf['targets']:
                print('target: ', target)
                self.cf['target'] = target

                # 2. Select variables
                data = excel_data[selected_Xs + [self.cf['target']]]

                # 3. Remove data if it contains 'nan'
                data = data[~data[self.cf['target']].isna()]
                if len(data) == 0:
                    continue

                # 4. Surfactant is converted to a numeric variable
                if 'Surfactant name' in data.columns.values:
                    data['Surfactant name'] = LabelEncoder().fit_transform(data['Surfactant name'])

                # 5. Normalization is performed
                scaled_df = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
                data = pd.DataFrame(scaled_df, index=data.index, columns=data.columns)

                # 6. Separate Xs and y
                df_X, df_y = data[selected_Xs], data[self.cf['target']]

                # 7. Convert y to a categorical variable for classification
                df_y = self.convert_to_categorical_variable(df_y, True)

                # 9. Perform several ML experiments
                sum_results = None
                all_results = {}
                for i in range(self.size_experiments):
                    # 9. Split into training and test part
                    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=.2)

                    # 13. Set data X and y for ML
                    self.ml.set_train_test_data(X_train, X_test, y_train, y_test)

                    # 14. Perform ML
                    results = self.ml.perform_ML()

                    if len(all_results) == 0:
                        all_results = {x: [v] for x, v in results.items()}
                    else:
                        for x, v in all_results.items():
                            for x2, v2 in results.items():
                                if x2 == x:
                                    v.append(v2)

                    if sum_results is None:
                        sum_results = results
                    else:
                        sum_results = {x: v + v2 for x, v in sum_results.items() for x2, v2 in results.items() if x2 == x}

                # 15. Set all results for the excel output
                for clf in self.ml.regressors:
                    name = type(clf).__name__
                    # self.excel[name + '-MAE'].append(avg_results[name])
                    self.excel[name + '-MAE'].append(round(mean(all_results[name]), 4))
                    self.excel[name + '-MAE-STD'].append(round(stdev(all_results[name]), 4))

                self.excel['Normalization'].append('True')
                self.excel['Train Size'].append(len(X_train))
                self.excel['Test Size'].append(len(X_test))
                self.excel["Surfactants"].append(surfactant)
                self.excel['Target'].append(target)
                self.excel['Regression'].append(self.cf['regression'])
                self.excel['X Removed'].append(X)
                self.excel['F Results'].append('')

        self.save_excel_file()

    def run_experiment(self, excel_data, surfactant='All'):
        for normalization in self.cf['normalization']:
            print('*****************************************************')
            print('normalization: ', normalization)
            for target in self.cf['targets']:
                print('target: ', target)
                self.cf['target'] = target

                # 2. Select variables
                data = excel_data[self.cf['selected_Xs'] + [self.cf['target']]]

                # 2.1 drop rows for nan values
                data = data.dropna()

                # 3. Remove data if it contains 'nan'
                data = data[~data[self.cf['target']].isna()]
                if len(data) == 0:
                    continue

                # 4. Surfactant is converted to a numeric variable
                if 'Surfactant name' in data.columns.values:
                    data['Surfactant name'] = LabelEncoder().fit_transform(data['Surfactant name'])

                # 9/5/2021 New process
                # 4.1. Convert a categorical variable to a discrete variable
                self.convert_to_discrete_variable(data)

                # 5. Normalization is performed
                if normalization is True:
                    scaled_df = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
                    data = pd.DataFrame(scaled_df, index=data.index, columns=data.columns)

                # 6. Separate Xs and y
                df_X, df_y = data[self.cf['selected_Xs']], data[self.cf['target']]

                # 7. Convert y to a categorical variable for classification
                df_y = self.convert_to_categorical_variable(df_y, normalization)

                # 8. Perform Sensitivity Analysis
                f_results = ''
                if self.cf['sensitivity analysis'] is True:
                    # columns = df_X.columns.copy()
                    # columns.append(self.cf['target'])
                    # self.show_heatmap_matrix(data, columns)
                    f_results = self.get_f_regression(df_X, df_y)
                    continue

                # 9. Perform several ML experiments
                all_results = {}
                for i in range(self.size_experiments):
                    # 9. Split into training and test part
                    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=.2)

                    # 13. Set data X and y for ML
                    self.ml.set_train_test_data(X_train, X_test, y_train, y_test)

                    # 14. Perform ML
                    results = self.ml.perform_ML()

                    if len(all_results) == 0:
                        all_results = {x: [v] for x, v in results.items()}
                    else:
                        for x, v in all_results.items():
                            for x2, v2 in results.items():
                                if x2 == x:
                                    v.append(v2)

                if self.cf['regression'] is True:
                    # 15. Set all results for the excel output
                    for clf in self.ml.regressors:
                        name = type(clf).__name__
                        # self.excel[name + '-MAE'].append(avg_results[name])
                        self.excel[name + '-MAE'].append(round(mean(all_results[name]), 4))
                        self.excel[name + '-MAE-STD'].append(round(stdev(all_results[name]), 4))
                else:
                    # 15. Set all results for the excel output
                    for clf in self.ml.classifiers:
                        name = type(clf).__name__
                        # self.excel[name + '-ACC'].append(avg_results[name])
                        self.excel[name + '-ACC'].append(round(mean(all_results[name]), 4))
                        self.excel[name + '-ACC-STD'].append(round(stdev(all_results[name]), 4))

                self.excel['Normalization'].append(normalization)
                self.excel['Train Size'].append(len(X_train))
                self.excel['Test Size'].append(len(X_test))
                self.excel["Surfactants"].append(surfactant)
                self.excel['Target'].append(target)
                self.excel['Regression'].append(self.cf['regression'])

    def run(self):
        # 1. Load data from xlsx
        excel_data = pd.read_excel(self.cf['file'], self.cf['sheet'])

        # Perform each target experiment
        for all_surfactants in self.cf['all_surfactants']:
            # Perform each target experiment
            if all_surfactants is True:
                self.run_experiment(excel_data, 'ALL')
            elif all_surfactants is False:
                df_surfactant = excel_data.groupby("Surfactant name")

                for s, df in df_surfactant:
                    self.run_experiment(df, s)

        self.save_excel_file()

ex = Experiment()

ex.run()
# ex.run_ml_removing_sensitivity_analysis()

winsound.Beep(1000, 440)