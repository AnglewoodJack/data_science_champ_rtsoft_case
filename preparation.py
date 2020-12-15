def data_preparation(file_list, lock_period, impute):
    '''
    recieves file list of data file names/paths in a certain order:
    1) icp das
    2) metering devices
    3) SVO
    4) VDNH
    5) COVID
    6) self-isolation index

    lock_period - can be specified as tuple (start date, edn date)in case new lockdown is introduced

    impute=True - NaN values will be imputed using KNN algorithm;
    impute=False - NaN values will be dropped
    '''
    # data processing and analysis
    import os
    import pandas as pd
    # module with information about holidays
    import holidays
    from processing import icp_preprocess, meter_preprocess, svo_preprocess, vdnh_preprocess
    from processing import isolation_preprocessing, covid_preprocessing, imputing_data

    # -------------------------------------------------DATA-LOAD--------------------------------------------------------

    # icp das
    icp_url = os.path.join(os.getcwd(), 'data', file_list[0])
    icp_features_url = os.path.join(os.getcwd(), 'data', 'building_features.pickle')
    # # metering device
    metering_device_url = os.path.join(os.getcwd(), 'data', file_list[1])
    metering_features_url = os.path.join(os.getcwd(), 'data', 'meter_features.pickle')
    # svo weather
    svo_weather_url = os.path.join(os.getcwd(), 'data', file_list[2])
    vdnh_weather_url = os.path.join(os.getcwd(), 'data', file_list[3])
    # covid cases Moscow
    covid_url = os.path.join(os.getcwd(), 'data', file_list[4])
    # isolation index
    isolation_url = os.path.join(os.getcwd(), 'data', file_list[5])

    # ---------------------------------------------FEATURE-SELECTION----------------------------------------------------

    # relevant icp_das features
    icp_das = icp_preprocess(icp_url, icp_features_url)
    # relevant metering devices features
    meter_dev = meter_preprocess(metering_device_url, metering_features_url)
    # temperature, atmospheric pressure, cloudness
    svo = svo_preprocess(svo_weather_url, ['T', 'U', 'c'])
    # precipitation
    vdnh = vdnh_preprocess(vdnh_weather_url)
    # covid cases
    cov = covid_preprocessing(covid_url)
    # isolation index
    iso = isolation_preprocessing(isolation_url)

    # ---------------------------------------------MERGING-DATASETS-----------------------------------------------------

    def merge_data(*args):
        '''
        merging datasets
        '''
        data = args[0]
        for i in range(1, len(args)):
            data = data.merge(args[i], how='left', on='time')

        return data

    data = merge_data(icp_das, meter_dev, svo, vdnh, cov, iso)
    data = data.set_index('time')

    # ----------------------------------------------ADD-COVID-CASES-----------------------------------------------------

    # populating daily values
    data['covid_cases'] = data['covid_cases'].groupby(pd.Grouper(freq='D')).ffill()
    data['isolation_idx'] = data['isolation_idx'].groupby(pd.Grouper(freq='D')).ffill()
    # fill leaking values
    data.loc[:'2020-03', 'covid_cases'] = data.loc[:'2020-03', 'covid_cases'].fillna(0)
    data.loc[:'2020-03','isolation_idx'] = data.loc[:'2020-03', 'isolation_idx'].fillna(0)

    # ----------------------------------------SPECIFY-WEEKDAYS-AND-MONTHS-----------------------------------------------

    # add weekday
    data['weekday'] = data.index.weekday
    # add month
    data['month'] = data.index.month
    # add yearday
    data['yearday'] = data.index.dayofyear
    # add monthday
    data['monthday'] = data.index.to_series().dt.day

    # -----------------------------------------------ADD-HOLIDAYS-------------------------------------------------------
    # add holidays
    rus_holidays = holidays.Russia()

    def holidays_selector(df, holidays_list):
        res = []
        for t in df.index:
            if t in holidays_list:
                res.append(1)
            else:
                res.append(0)
        return pd.DataFrame({'time': df.index,  'holiday': res})

    all_holidays = holidays_selector(data, rus_holidays)

    # -----------------------------------------------ADD-LOCKDOWN-------------------------------------------------------

    # set time of lockdown in Moscow
    lockdown = pd.DataFrame(pd.date_range(start='2020-03-30 00:00',
                                          end='2020-06-08 23:00', freq='H'), columns=['time'])
    # set corresponding column to 1
    lockdown['lockdown'] = 1

    # in case of new lockdown
    if lock_period is not None:
        new_lockdown = pd.DataFrame(pd.date_range(start=lock_period[0],
                                              end=lock_period[1], freq='H'), columns=['time'])

        lockdown.append(new_lockdown)

    # add lockdown periods
    data = merge_data(data, all_holidays, lockdown).set_index('time')

    # -----------------------------------------------FILL-NAs-----------------------------------------------------------

    data['lockdown'] = data['lockdown'].fillna(0)
    data['precipitation'] = data['precipitation'].fillna(0)

    if impute:
        # TODO: make user to decide which columns to impute
        data = imputing_data(data)

    return data
