def app_predict(file_list, lock_period=None):
	'''

	'''
	# ---------------------------------------------IMPORT-PACKAGES------------------------------------------------------
	import pickle
	import pandas as pd
	import numpy as np
	from app_preparation import app_data_preparation
	from datetime import timedelta

	# -----------------------------------------------LOAD-DATA----------------------------------------------------------
	# load previous dataset
	with open('model.pickle', 'rb') as fp:
		data, xgb, gbr = pickle.load(fp)

	# load update
	update = app_data_preparation(file_list, lock_period, impute=False)

	# concatenate datasets
	data = data.append(update)

	# ------------------------------------------MAKE-TEST-DATASET-------------------------------------------------------

	# make time period for the next day
	end_date = (data.index[-1] + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')
	start_date = (data.index[-1] + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')
	temp = pd.DataFrame(pd.date_range(start=start_date,end=end_date, freq='H'), columns=['time'])
	# convert to pandas datetime format
	temp = temp.apply(lambda row: pd.to_datetime(row), axis=0).set_index('time')

	# concatenate datasets
	df = data.append(temp)
	# shift data by 24 hour for prediction
	df.iloc[:, 1:] = df.iloc[:, 1:].shift(24)
	# start date index
	inx = df.index.get_loc(start_date)
	# slice out test dataset
	x_test = df.iloc[inx:, 1:]

	# -----------------------------------------MAKE-PREDICTION----------------------------------------------------------

	y_xgb = xgb.predict(x_test)
	y_gbr = gbr.predict(x_test)

	y_pred = (y_xgb + y_gbr)/2

	return np.array(x_test.index), y_pred






