import streamlit as st
import pandas as pd
import plotly.io as pio
import matplotlib.pyplot as plt
from app_predict import app_predict
from app_fit import app_fit

# set title
st.title('Прогнозирование нагрузки офисных задний в условиях пандемии COVID-19')

# white backgraound
pio.templates.default = 'plotly_white'
# create sidebar
st.sidebar.subheader('Панель управления')
# file uploader boxes
bd_test = st.sidebar.file_uploader('файл с данными ICP DAS')
bdm_test = st.sidebar.file_uploader('файл с данными счётчиков')
svo_test = st.sidebar.file_uploader('файл с данными погоды Шереметьево')
vdnh_test = st.sidebar.file_uploader('файл с данными погоды ВДНХ')
covid_test = st.sidebar.file_uploader('файл с данными по COVID')
iso = st.sidebar.file_uploader('файл с данными по индексу самоизоляции')
# make list of files
file_list = [bd_test, bdm_test, svo_test, vdnh_test, covid_test, iso]
# fit model button
submit_fit = st.button('обучить модель')
# predict button
submit_prediction = st.button('предсказать')
# data visualization selection
choice = st.radio('Отображение', options=['таблица', 'график'])


def make_plot(df):
	'''
	create plot from predicted data
	'''
	fig, ax = plt.subplots(figsize=(20,10))
	ax.plot(df[0], df[1], color='orange', lable='Прогноз')
	plt.xlabel('Дата/время')
	plt.ylabel('кВтч')
	plt.xlim(min(df[0]), max(df[0]))
	plt.ylim(min(df[1]), max(df[1]))
	st.pyplot()

def make_prediction():
	'''
	run prediction
	'''
	if bd_test and bdm_test and svo_test and vdnh_test and covid_test and iso:
		if submit_prediction:
			result = app_predict(file_list)
			df = pd.DataFrame({'Дата-время': result[0], 'Потребление эл-ва': result[1]})
			return df
		else:
			st.write('запустите прогноз')

	else:
		st.write('загрузите данные')

st.subheader('Результат работы алгоритма')

# show prediction data
prediction = make_prediction()
if prediction:
	if choice == 'график':
		make_plot(prediction)
		# st.line_chart(prediction.set_index('Дата-время'))
	else:
		st.dataframe(prediction, width=1000, height=10000)

# run model fit
if bd_test and bdm_test and svo_test and vdnh_test and covid_test and submit_fit:
	app_fit(file_list)
