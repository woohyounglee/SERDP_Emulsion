import numpy as np
import pandas as pd
from regressionML import RegressionML
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statistics import stdev
from statistics import mean
import winsound
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from model_setting import Model_Setting


class SensitivityAnalysisExperiment(Model_Setting):
    def __init__(self, regression=True):
        super().__init__()

        # self.cf['targets'] = ['OV (Oily value)']
        # self.cf['targets'] = ['OS (Oil separation)']
        self.cf['targets'] = ['Turbidity']

        self.size_experiments = 10
        self.ml = RegressionML()
        for clf in self.ml.regressors:
            name = type(clf).__name__
            self.excel[name + '-MAE'] = []

        for clf in self.ml.regressors:
            name = type(clf).__name__
            self.excel[name + '-MAE-STD'] = []

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

    def run_ml_removing_sensitivity_analysis(self):
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
                # df_y = self.convert_to_categorical_variable(df_y)

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

                self.excel['Train Size'].append(len(X_train))
                self.excel['Test Size'].append(len(X_test))
                self.excel['Target'].append(target)
                self.excel['X Removed'].append(X)
                self.excel['F Results'].append('')

        self.save_excel_file()

    def run_experiment(self, excel_data):
        for target in self.cf['targets']:
            print('target: ', target)
            self.cf['target'] = target

            # 2. Select variables
            data = excel_data[self.cf['selected_Xs'] + [self.cf['target']]]

            # 3. Remove data if it contains 'nan'
            data = data[~data[self.cf['target']].isna()]

            # 5. Normalization is performed
            scaled_df = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
            data = pd.DataFrame(scaled_df, index=data.index, columns=data.columns)

            # 6. Separate Xs and y
            df_X, df_y = data[self.cf['selected_Xs']], data[self.cf['target']]

            # 8. Perform Sensitivity Analysis
            f_results = ''
            # columns = df_X.columns.copy()
            # columns.append(self.cf['target'])
            # self.show_heatmap_matrix(data, columns)
            f_results = self.get_f_regression(df_X, df_y)

    def run(self):
        # 1. Load data from xlsx
        excel_data = pd.read_excel(self.cf['file'], self.cf['sheet'])

        self.run_experiment(excel_data)

        self.save_excel_file()

# 1. looks at F-score
ex = SensitivityAnalysisExperiment().run()

# 2. performs one variable removal test
# SensitivityAnalysisExperiment().run_ml_removing_sensitivity_analysis()

winsound.Beep(1000, 440)