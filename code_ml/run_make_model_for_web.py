import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle
from model_setting import Model_Setting


class Make_Model(Model_Setting):
    def __init__(self, regression=True):
        super().__init__(regression)

    def run_make_model(self, excel_data, surfactant='All'):
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
                self.ml.set_train_test_data(df_X, None, df_y, None)

                if self.cf['regression'] is True:
                    ml = self.ml.train_ML(ml_name='RandomForestRegressor')
                    with open(f'../web/ml_models/ML_Alg_{target}_Regressor.pkl', 'wb') as fid:
                        pickle.dump(ml, fid)
                else:
                    ml = self.ml.train_ML(ml_name='RandomForestClassifier')
                    with open(f'../web/ml_models/ML_Alg_{target}_Classifier.pkl', 'wb') as fid:
                            pickle.dump(ml, fid)

    def run(self):
        # 1. Load data from xlsx
        excel_data = pd.read_excel(self.cf['file'], self.cf['sheet'])
        self.run_make_model(excel_data, 'ALL')

ex = Make_Model(regression=True)
ex.run()

ex = Make_Model(regression=False)
ex.run()
