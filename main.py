from fit import fit
from predict import predict
import os

# path to train files
bd_train = os.path.join('train', 'building_data_train.xlsx')
bdm_train = os.path.join('train', 'building_data_meters_train.xlsx')
svo_train = os.path.join('train', 'SVO_train.xls')
vdnh_train = os.path.join('train', 'VDNH_train.xls')
covid_train = os.path.join('train', 'covid_train.xlsx')
iso_train = os.path.join('train', 'isolation_train.xlsx')


# remove hashtags below to fit the model
# file_list = [bd_train, bdm_train,svo_train, vdnh_train, covid_train, iso_train]
# model learning
# fit(file_list)

# path to test files
bd_test = os.path.join('test', 'data_test_0.xlsx')
bdm_test = os.path.join('test', 'building_data_meters_test.xlsx')
svo_test = os.path.join('test', 'SVO_test.xls')
vdnh_test = os.path.join('test', 'VDNH_test.xls')
covid_test = os.path.join('test', 'covid_test.xlsx')
covid_test = os.path.join('test', 'covid_test.xlsx')
iso_test = os.path.join('test', 'isolation_test.xlsx')

file_list = [bd_test, bdm_test, svo_test, vdnh_test, covid_test, iso_test]

# model prediction
result = predict(file_list)

# printing results
print(result)
