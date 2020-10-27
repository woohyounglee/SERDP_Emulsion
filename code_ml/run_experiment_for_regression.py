import pandas as pd
from regressionML import RegressionML
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from statistics import stdev
from statistics import mean
import winsound
from run_experiment import Experiment


class RegressionExperiment(Experiment):
    def __init__(self, regression=True):
        super().__init__()

        self.cf['targets'] = ['OV (Oily value)']
        # self.cf['targets'] = ['OS (Oil separation)']
        # self.cf['targets'] = ['Turbidity']

        self.normalization = False
        # self.normalization = True

        self.ml = RegressionML()
        for clf in self.ml.regressors:
            name = type(clf).__name__
            self.excel[name + '-MAE'] = []

    def run_experiment(self, excel_data):
        for target in self.cf['targets']:
            print('target: ', target)
            self.cf['target'] = target

            # 2. Select variables
            data = excel_data[self.cf['selected_Xs'] + [self.cf['target']]]

            # 3. Remove data if it contains 'nan'
            data = data[~data[self.cf['target']].isna()]

            # 4. Normalization for the target variable
            scaled_target_df = MinMaxScaler(feature_range=(0, 1)).fit_transform(data[self.cf['target']].to_frame())
            data[self.cf['target']] = scaled_target_df

            # 5. Normalization is performed
            if self.normalization is True:
                scaled_df = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
                data = pd.DataFrame(scaled_df, index=data.index, columns=data.columns)

            # 6. Separate Xs and y
            df_X, df_y = data[self.cf['selected_Xs']], data[self.cf['target']]

            # 9. Split into training and test part
            X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=.2, random_state=42)

            # 10. Split data for cross validation
            cv_results = {}
            n_splits = self.cf['n_splits']
            if n_splits > 0:
                kf = KFold(n_splits=n_splits)

                for train_index, val_index in kf.split(X_train):
                    train_x, val_x = X_train.iloc[train_index], X_train.iloc[val_index]
                    df_y_train = pd.DataFrame(data=y_train)
                    train_y, val_y = df_y_train.iloc[train_index], df_y_train.iloc[val_index]

                    # 11. Set data X and y for cross validation
                    self.ml.set_train_test_data(train_x, val_x, train_y, val_y)

                    # 12. Perform ML for cross validation
                    results = self.ml.perform_ML()

                    if len(cv_results) == 0:
                        for x, v in results.items():
                            cv_results[x] = []

                    for x, v in results.items():
                        cv_results[x].append(v)

                for x, v in cv_results.items():
                    print(f'[{x}] mean-std [{round(mean(v), 4)} ({round(stdev(v), 4)})')

            # 13. Set data X and y for ML
            self.ml.set_train_test_data(X_train, X_test, y_train, y_test)

            # 14. Perform ML
            results = self.ml.perform_ML(file_save=f'../output/{target}')

            # 15. Set all results for the excel output
            for clf in self.ml.regressors:
                name = type(clf).__name__
                self.excel[name + '-MAE'].append(round(results[name], 4))
                print(name, round(results[name], 4))

            self.excel['Train Size'].append(len(X_train))
            self.excel['Test Size'].append(len(X_test))
            self.excel['Target'].append(target)

    def run(self):
        # 1. Load data from xlsx
        excel_data = pd.read_excel(self.cf['file'], self.cf['sheet'])

        self.run_experiment(excel_data)

        self.save_excel_file()


ex = RegressionExperiment().run()

winsound.Beep(1000, 440)
