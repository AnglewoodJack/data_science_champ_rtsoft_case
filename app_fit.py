def app_fit(file_list, lock_period=None, impute=False):
	'''
	process data and fit model
	impute specified as False for current use case
	'''
	# ---------------------------------------------IMPORT-PACKAGES------------------------------------------------------
	import pickle
	import numpy as np
	from math import floor

	from app_preparation import app_data_preparation
	# data markup
	from sklearn.model_selection import TimeSeriesSplit
	from sklearn.model_selection import GridSearchCV
	# metrics
	import sklearn.metrics as metrics
	# models
	from sklearn.ensemble import GradientBoostingRegressor
	from xgboost import XGBRegressor

	# ------------------------------------------------GET-DATA----------------------------------------------------------

	# get processed dataset
	data = app_data_preparation(file_list, lock_period, impute)
	# all data 24-hour shift
	data.iloc[:, 1:] = data.iloc[:, 1:].shift(24)
	# remove NAs
	data.dropna(inplace=True)

	# ------------------------------------------------METRICS-----------------------------------------------------------

	# create RMSE metric
	def rmse(actual, predict):
		predict = np.array(predict)
		actual = np.array(actual)
		distance = predict - actual
		square_distance = distance ** 2
		mean_square_distance = square_distance.mean()
		score = np.sqrt(mean_square_distance)
		return score

	# RMSE score
	rmse_score = metrics.make_scorer(rmse, greater_is_better = False)

	# -----------------------------------------------MODELING-----------------------------------------------------------

	# time series cross-validation
	month_num = ((data.index[-1] - data.index[0])/np.timedelta64(1, 'M'))  # number of months between two dates
	tscv = TimeSeriesSplit(n_splits=floor(month_num))

	# split into target and features variables
	x_train = data.drop(['target'], axis = 1)
	y_train = data.loc[:, 'target']

	# --------------------------------------------XGB-REGRESSOR---------------------------------------------------------

	# select parameters for XGB
	xgb = XGBRegressor(random_state=0)
	xgb_param_search = {
		'n_estimators': [10, 20, 30, 400],
		'max_features': ['auto'],
		'max_depth' : [i for i in range(3,7)]
	}

	xgb_gsearch = GridSearchCV(estimator=xgb, cv=tscv, param_grid=xgb_param_search, scoring = rmse_score)
	xgb_gsearch.fit(x_train, y_train)

	# -------------------------------------------GBR-REGRESSOR----------------------------------------------------------

	# select parameters for GBR
	gbr = GradientBoostingRegressor(random_state=0)
	gbr_param_search = {
		'n_estimators': [10, 20, 30, 400],
		'max_features': ['auto'],
		'max_depth' : [i for i in range(3,7)]
	}

	gbr_gsearch = GridSearchCV(estimator=gbr, cv=tscv, param_grid=gbr_param_search, scoring = rmse_score)
	gbr_gsearch.fit(x_train, y_train)

	# -------------------------------------------SAVE-TO-PICKLE----------------------------------------------------------

	with open('model.pickle', 'wb') as fp:
		pickle.dump((data, xgb_gsearch.best_estimator_, gbr_gsearch.best_estimator_), fp)
