import pandas as pd
from model_setting import Model_Setting
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Run_Experiments(Model_Setting):
    def __init__(self, regression=True):
        super().__init__(regression)

    def run_make_model(self, excel_data):
        for normalization in self.cf['normalization']:
            print('*****************************************************')
            print('normalization: ', normalization)

            for target in self.cf['targets']:
                print('*****************************************************')
                print('target: ', target)
                self.cf['target'] = target

                # 2. Select variables
                data = excel_data[self.cf['selected_Xs'] + [self.cf['target']]]

                # 2.1 Normalzie the target variable
                data[self.cf['target']] = data[self.cf['target']] / data[self.cf['target']].max()

                # 3. Remove data if it contains 'nan'
                data = data[~data[self.cf['target']].isna()]
                if len(data) == 0:
                    continue

                # 4. Surfactant is converted to a numeric variable
                data['Surfactant name'] = LabelEncoder().fit_transform(data['Surfactant name'])

                # 5. Normalization is performed
                if normalization is True:
                    scaled_df = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
                    data = pd.DataFrame(scaled_df, index=data.index, columns=data.columns)

                # 6. Separate Xs and y
                df_X, df_y = data[self.cf['selected_Xs']], data[self.cf['target']]

                # 7. Convert y to a categorical variable for classification
                df_y = self.convert_to_categorical_variable(df_y, normalization)

                # 8. Split into training and test part
                X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=.2, random_state=42)

                # 9. Set data X and y for ML
                self.ml.set_train_test_data(X_train, X_test, y_train, y_test)

                # 10. Perform ML
                self.ml.perform_ML()

                if self.cf['regression'] is True:
                    mls = self.ml.regressors
                else:
                    mls = self.ml.classifiers

                for clf in mls:
                    name = type(clf).__name__
                    model = clf.fit(self.ml.X_train, self.ml.y_train)
                    predicted = clf.predict(self.ml.X_test)
                    print('predicted', predicted)
                    print('actual', y_test)

    def run(self):
        # 1. Load data from xlsx
        excel_data = pd.read_excel(self.cf['file'], self.cf['sheet'])
        self.run_make_model(excel_data)


ex = Run_Experiments(regression=True)
ex.run()
