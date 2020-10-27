import numpy as np
import pandas as pd
from classificationML import ClassificationML
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import winsound
from imblearn.over_sampling import SMOTE
from run_experiment import Experiment
from statistics import stdev
from statistics import mean


class ClassificationExperiment(Experiment):
    def __init__(self):
        super().__init__()

        self.cf['targets'] = ['OV (Oily value)']
        # self.cf['targets'] = ['OS (Oil separation)']
        # self.cf['targets'] = ['Turbidity']

        self.ml = ClassificationML()
        for clf in self.ml.classifiers:
            name = type(clf).__name__
            self.excel[name + '-ACC'] = []

    def convert_to_categorical_variable(self, df_y):
        # trisection: 3 equal parts
        bins = [np.NINF, 0.33, 0.66, np.inf]
        # bins = [np.NINF, 0.2, 0.80, np.inf]
        # names = ['low', 'mid', 'high']
        names = ['0', '1', '2']

        # bisection: 2 equal parts
        # bins = [np.NINF, 0.5, np.inf]
        # names = ['low', 'high']

        df_y = pd.cut(df_y, bins, labels=names)

        return np.ravel(df_y)

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

            # 7. Convert y to a categorical variable for classification
            df_y = self.convert_to_categorical_variable(df_y)

            smote = SMOTE()
            df_X, df_y = smote.fit_sample(df_X, df_y)

            # 9. Split into training and test part
            X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=.2, random_state=42, stratify=df_y)

            # Save the selected data as csv files
            # X_train.reset_index(drop=True, inplace=True)
            # X_test.reset_index(drop=True, inplace=True)
            # all_train = pd.concat([X_train, pd.Series(y_train)], axis=1, ignore_index=True)
            # all_test = pd.concat([X_test, pd.Series(y_test)], axis=1, ignore_index=True)
            # all_train.to_csv('temp_data/train.csv', index=False)
            # all_test.to_csv('temp_data/eval.csv', index=False)

            # 10. Split data for cross validation
            cv_results = {}
            n_splits = self.cf['n_splits']
            if n_splits > 0:
                kf = KFold(n_splits=n_splits)

                sum_cv_results = None
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
            results = self.ml.perform_ML()

            # 15. Set all results for the excel output
            for clf in self.ml.classifiers:
                name = type(clf).__name__
                self.excel[name + '-ACC'].append(round(results[name], 4))
                print(name, round(results[name], 4))

            self.excel['Train Size'].append(len(X_train))
            self.excel['Test Size'].append(len(X_test))
            self.excel['Target'].append(target)

    def run(self):
        # 1. Load data
        excel_data = pd.read_excel(self.cf['file'], self.cf['sheet'])
        self.run_experiment(excel_data)
        self.save_excel_file()


ClassificationExperiment().run()
winsound.Beep(1000, 440)