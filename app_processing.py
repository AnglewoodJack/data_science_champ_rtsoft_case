def app_icp_preprocess(data_url, selected_features_url):
    '''
    ICP DAS data preprocessing
    '''
    import pickle
    import pandas as pd
    # load data
    df = pd.read_excel(data_url)
    # local time shift - UTC+3
    df['time'] = df['time'] + pd.DateOffset(hours=3)
    # target column
    df['target'] = (df['_Bus_1_p_TM1H_mv_val'] + df['_Bus_2_p_TM1H_mv_val'] +
                    df['_Bus_3_p_TM1H_mv_val'] + df['_Bus_4_p_TM1H_mv_val'])

    with open(selected_features_url, 'rb') as fp:
        # load list of features
        columns = pickle.load(fp)

    # keep selected features
    df = df[['time', 'target'] + columns]

    # sorting by time
    df = df.sort_values(by='time', ascending=True).set_index('time')

    return df


def app_meter_preprocess(data_url, selected_features_url):
    '''
    metering devices data preprocessing
    '''
    import pickle
    import pandas as pd
    # load data
    df = pd.read_excel(data_url)
    # local time shift - UTC+3
    df['time'] = df['time'] + pd.DateOffset(hours=3)

    with open(selected_features_url, 'rb') as fp:
        # load list of features
        columns = pickle.load(fp)

    # keep selected features
    df = df[['time'] + columns]

    # average data by hour
    df = df.set_index('time').resample('H').mean()
    df.reset_index(drop=True)

    def remove_anomaly(df):
        '''
		remove anomalies
		'''
        for i in range(1,5):
            median = df[f'_electricityMeter_{i}_p_T30M_mv_val'].median()
            df[f'_electricityMeter_{i}_p_T30M_mv_val'] = df.apply(lambda row: median if
                row[f'_electricityMeter_{i}_p_T30M_mv_val']>500 else
                row[f'_electricityMeter_{i}_p_T30M_mv_val'], axis=1)
        return df

    return remove_anomaly(df)


def app_svo_preprocess(data_url, selected_features_url):
    '''
    wheather data preprocessing
    '''
    import pandas as pd
    df = pd.read_excel(data_url, header=6)
    # rename time column and convert to datetime format
    df = df.rename(columns={'Местное время в Шереметьево / им. А. С. Пушкина (аэропорт)': 'time'})
    df['time'] = pd.to_datetime(df['time'])
    # split cloudness variable into to subparameters
    df['c_1'] = df['c'].str.extract('(\d+)\s\w')
    df['c_2'] = df['c'].str.extract('([\w\s]+)')
    # comnbine data on cloudness
    df['c'] = df.apply(lambda row: 0 if row['c_2'] ==
                       'Нет существенной облачности' else float(row['c_1']), axis=1)
    # convert temperature into Kelvins to get rid of negative values
    df['T'] = df['T'] + 273.15

    #### keep only temperature, atmospheric pressure and cloudness columns in selected features ###

    df = df[['time'] + selected_features_url]

    return df


def app_vdnh_preprocess(data_url):
    '''
    wheather data preprocessing (get precipitation)
    '''
    import pandas as pd
    df = pd.read_excel(data_url, header=6)
    # rename time column and convert to datetime format
    df = df.rename(columns={'Местное время в Москве (ВДНХ)': 'time'})
    df['time'] = pd.to_datetime(df['time'])
    # comnbine data on cloudness
    df['precipitation'] = df.apply(lambda row: 0 if (row['RRR'] == 'Следы осадков' or
                                                     row['RRR'] == 'Осадков нет')
                                   else float(row['RRR']), axis=1)

    df['precipitation'] = df['precipitation'].fillna(0)

    df = df[['time', 'precipitation']]

    return df


def app_covid_preprocessing(data_url):
    '''
    covid data preprocessing
    '''
    import pandas as pd
    df = pd.read_excel(data_url, header=None)
    # convert to datetime
    df[0] = pd.to_datetime(df[0])
    df = df.rename(columns={0: 'time', 1: 'covid_cases'})

    return df


def app_isolation_preprocessing(data_url):
    '''
    isolation index preprocessing
    '''
    import pandas as pd
    df = pd.read_excel(data_url, header=0)
    df = df.rename(columns={'DateTime':'time', 'Индекс':'isolation_idx'})
    # convert to datetime
    df['time'] = pd.to_datetime(df['time'])

    return df

def app_imputing_data(df):
    '''
    imputes data using KNN method
    '''
    from sklearn.impute import KNNImputer
    import pandas as pd
    # impute missing values using KNN method
    imputer = KNNImputer()
    df_trans = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(data = df_trans, columns = df.columns)
    df_imputed['time'] = df.index
    df_imputed = df_imputed.set_index('time')

    return df_imputed