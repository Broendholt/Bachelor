import pandas as pd
from sklearn.feature_selection import *
import DataPrep as dp
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("start")
    #
    # data = pd.read_excel('output.xlsx')
    # d, x_1, x_2, y = dp.prepare_data_feature_selection(data, 0, 3700, 1, 1)
    #
    # # data = pd.read_excel('output.xlsx')
    # # d, x_1, x_2, y = dp.prepare_data_feature_selection(data, 13750, 40000, 1, 1)
    # # d, x_1, x_2, y = dp.prepare_data_feature_selection(data, 13750, 86190, 1, 1)
    #
    #
    #
    # print(len(y), y.size)
    # print(len(x_1), x_1.size)
    # print(len(x_2), x_2.size)
    #
    # print('Longitude')
    # print('Univariate feature selection')
    # univariate_selection(y, x_1, 10)
    # print()
    # print('-------------------------------------------')
    # print('Latitude')
    # print('Univariate feature selection')
    # univariate_selection(y, x_2, 10)
    #
    # print()
    # print('-------------------------------------------')
    #
    # print('Longitude')
    # print('feature importance')
    # feature_importance(y, x_1, 10)
    # print()
    # print('-------------------------------------------')
    # print('Latitude')
    # print('feature importance')
    # feature_importance(y, x_2, 10)


# Univariate Selection
def univariate_selection(independent_columns, target_columns, amount):

    # apply SelectKBest class to extract top 10 best features
    best_features = SelectKBest(score_func=mutual_info_regression, k=amount)
    fit = best_features.fit(independent_columns, target_columns)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(independent_columns.columns)
    # concat two dataframes for better visualization
    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(feature_scores.nlargest(amount, 'Score'))  # print 10 best features


def feature_importance(independent_columns, target_columns, amount):
    independent_columns = independent_columns.astype('int64')
    target_columns = target_columns.astype('int64')

    model = ExtraTreesClassifier(n_estimators=100, random_state=0)
    model.fit(independent_columns, target_columns)
    # print(model.feature_importances_)

    # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=independent_columns.columns)
    feat_importances.nlargest(amount).plot(kind='barh')
    plt.show()


def get_feature_selection_rows(data_set, top):

    if data_set == 1:
        lat_features = np.array(['WindSpeedTrue', 'Azipot4', 'Shaft3', 'Azipot3',
                         'SOG', 'Shaft4', 'HDG', 'Shaft2', 'WindSpeedRel', 'Azipot1'])

        lon_features = np.array(['WindSpeedTrue', 'Azipot1', 'DfAzipot1', 'Shaft3',
                         'Azipot4', 'SOG', 'Azipot3', 'Shaft4', 'Shaft2', 'Shaft1'])

        return lon_features[:top], lat_features[:top]

    elif data_set == 2:
        lat_features = np.array(['SOG', 'STW_UW', 'SOG_U', 'PD_C', 'W_Speed_R_kn', 'CentRPM_S_abs',
                                 'CentRPM_S_pct', 'CentRPM_C', 'depth', 'PortRPM_S_pct'])

        lon_features = np.array(['SOG', 'STW_UW', 'SOG_U', 'W_Speed_R_kn', 'Heading', 'PD_C',
                                 'W_Dir_R', 'depth', 'CentRPM_S_abs', 'CentRPM_S_pct'])

        return lon_features[:top], lat_features[:top]

main()

