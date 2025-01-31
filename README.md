# data_science_champ_rtsoft_case
Решение задачи от RTSoft "Прогноз нагрузки на электросеть", занявшее третье место на Чемпионате Data Science Университета 2035, Платформы НТИ и Правительства Москвы.

## Описание задачи
**Прогноз нагрузки на электросеть**

Разработка модели машинного обучения, позволяющей спрогнозировать
нагрузки на энергосеть с целью выявления оптимальных режимов потребления
электроэнергии офисного здания в условиях пандемии COVID-19

**1. Ознакомьтесь с описанием задачи**

В связи с пандемией COVID-19, переход ряда сотрудников на удаленную
работу приобрел стохастический характер, \
что требует совершенствования используемых подходов к прогнозированию
потребления электрической энергии. \
Представляется целесообразным учет дополнительных параметров для
прогнозирования. \
В настоящее время стремительно развиваются системы управления
энергопотреблением (EMS System), позволяющие снизить плату за
электрическую энергию за счет оптимального управления системой
электроснабжения здания.\
В офисном здании установлена система накопления электрической энергии,
для формирования оптимальных управляющих воздействий на которую
требуется выполнение качественного прогнозирования электрической
нагрузки. \
Это позволяет задействовать накопители электрической энергии в нужные
периоды времени для снижения потребляемой из сети электрической энергии
в часы пиковой нагрузки (при использовании двухставочного тарифа) или в
пиковой зоне суток (при использовании дифференцированного тарифа на
электрическую энергию). \
Более высокое качество прогноза потребления электрической энергии
позволяет более эффективным образом задействовать накопитель
электрической энергии, а, следовательно, получать большую выгоду от его
использования.\
В рамках задания необходимо подготовить модель машинного обучения, \
обеспечивающую высокую точность прогнозирования нагрузки офисного здания
в условиях пандемии COVID-19. \
\
**2. Проблема**\
\
В связи с пандемией COVID-19, переход ряда сотрудников на удаленную
работу приобрел стохастический характер,\
что требует совершенствования используемых подходов к прогнозированию
потребления электрической энергии. \
В связи с этим представляется целесообразным учет дополнительных
параметров для прогнозирования.\
\
\
**3. Исходные данные**\
\
В качестве исходных данных предоставляются следующие данные:

1.  Датасет building_data.xlsx (данные с датчиков ICP DAS)

2.  Датасет building_data_meters.xlsx (данные с приборов учета)

3.  Датасет с метеоданными можно загрузить с сайта rp5.ru . 

В датасете представлено время UTC.\
\
**4. Формулировка задания**

1.  Выполнить очистку данных от аномалий (Anomaly detection);

2.  Произвести кластеризацию данных (выявить выходные и праздничные дни,
    а также дни в которые большая часть сотрудников находилась на
    удаленной работе). 

3.  Построить и обучить модель прогнозирования, которая позволяет
    формировать почасовой прогноз на завтрашний день (прогноз
    формируется в 00-00 часов).

4.  Обеспечить ежечасную актуализацию прогноза потребления нагрузки на
    основании вновь поступающих данных \
    (например, в 13-30 уточняется прогноз на часы начиная с 13-го и
    заканчивая 23-м).

5.  При оценке качества прогноза необходимо игнорировать ночные часы (от
    23-00 до 6-00 мск).

6.  Выявить дополнительные корреляции с другими параметрами (погодные
    условия, информация о карантине, загруженность дорог и пр.). \
    Погодные данные можно получить на сайте weather.com.

7.  Оценить влияние учета дополнительной информации на качество
    прогнозирования.\
    Используемые библиотеки - на усмотрение участников 

> **Результат должен быть представлен в формате Jupyter Notebook, язык
> программирования Python. **\
> \
> Допускается применение другой среды разработки по согласованию с
> постановщиком задачи.

[[Архив погоды (ВДНХ) - 6 измерений в
день]{.ul}](https://rp5.ru/%D0%90%D1%80%D1%85%D0%B8%D0%B2_%D0%BF%D0%BE%D0%B3%D0%BE%D0%B4%D1%8B_%D0%B2_%D0%9C%D0%BE%D1%81%D0%BA%D0%B2%D0%B5_(%D0%92%D0%94%D0%9D%D0%A5)) \
\
[[архив погоды (Шереметьево) - измерения каждые
полчаса]{.ul}](https://rp5.ru/%D0%90%D1%80%D1%85%D0%B8%D0%B2_%D0%BF%D0%BE%D0%B3%D0%BE%D0%B4%D1%8B_%D0%B2_%D0%A8%D0%B5%D1%80%D0%B5%D0%BC%D0%B5%D1%82%D1%8C%D0%B5%D0%B2%D0%BE,_%D0%B8%D0%BC._%D0%90._%D0%A1._%D0%9F%D1%83%D1%88%D0%BA%D0%B8%D0%BD%D0%B0_(%D0%B0%D1%8D%D1%80%D0%BE%D0%BF%D0%BE%D1%80%D1%82),_METAR)\
\
[[Данные для скачивания находятся
здесь]{.ul}](https://files.2035.university/f/db927d3080eb43b487b4/)

Прогнозируемая величина, по качеству которой мы будем оценивать
результат - это суммарная почасовая неуправляемая нагрузка (активная
мощность).\
Она является суммой неуправляемых
нагрузок **(\_Bus_i\_p_TM1H_mv_val)** по четырем вводам в здание:
```
_Bus_1_p_TM1H_mv_val + _Bus_2_p_TM1H_mv_val + _Bus_3_p_TM1H_mv_val + _Bus_4_p_TM1H_mv_val
```
\
Это то же самое, что сумма суммарных
потреблений **(\_externalGrid_Bus_i\_p_TM1H_mv_val)** по всем вводам
минус потребление всех управляемых крыш плюс мощность накопителей
```
(_externalGrid_Bus_1_p_TM1H_mv_val + _externalGrid_Bus_2_p_TM1H_mv_val
+ _externalGrid_Bus_3_p_TM1H_mv_val + _externalGrid_Bus_4_p_TM1H_mv_val)
- (_roof_1_Bus_4_p_TM1H_mv_val + _roof_2_Bus_4_p_TM1H_mv_val + _roof_Bus_1_p_TM1H_mv_val)
+ (_Accumulator_Bus_1_p_TM1H_mv_val + _Accumulator_Bus_2_p_TM1H_mv_val
+ _Accumulator_Bus_3_p_TM1H_mv_val + _Accumulator_Bus_4_p_TM1H_mv_val)
```

Датасет building_data_meters содержит измерения коммерческого учета
электроэнергии суммарных нагрузок по вводам **(1,2,3,4)**,\
без разделения на управляемую или нет. \
Использовать данные из этого датасета или нет - на ваше усмотрение,
однако обращу внимание, что иногда данные в датасете building_data
являются недостоверными или отсутствуют, при этом в датасете
building_data_meters за то же время данные есть. \
Также в building_data_meters есть данные повышенной дискретности - 3
минуты.\
Мы будем оценивать качество прогноза именно суммы всех неуправляемых
нагрузок, однако в датасете есть значения индвидуально по вводам и по
фазам вводов, возможно это поможет при прогнозировании. \
\
Для пункта 3) задания необходимо формировать почасовой прогноз на
следующий день. Т.е. каждый день ровно в **00:00** необходимо определить
24 числа - значения потребления на следующий день. \
При этом есть информация о потреблении за все предыдущие дни, но
очевидно нет информации из будущего. Метрику определим, как среднее RMSE
за все дневные часы в прогнозном интервале (т.е. с **7:00 до 22:00
МСК**). \
\
Прогноз ночного потребления нам неинтересен. \
Обратите внимание, в датасете время в UTC.

## Дополинтельные уточнеиня
Прогнозируемая величина, по качеству которой мы будем оценивать
результат - это суммарная почасовая неуправляемая нагрузка (активная
мощность). Она является суммой неуправляемых нагрузок
(**\_Bus_i\_p_TM1H_mv_val**) по четырем вводам в здание:

```
_Bus_1_p_TM1H_mv_val + _Bus_2_p_TM1H_mv_val + _Bus_3_p_TM1H_mv_val + _Bus_4_p_TM1H_mv_val
```

Это то же самое, что сумма суммарных потреблений
(**\_externalGrid_Bus_i\_p_TM1H_mv_val**) по всем вводам минус
потребление всех управляемых крыш плюс мощность накопителей

```
(_externalGrid_Bus_1_p_TM1H_mv_val + _externalGrid_Bus_2_p_TM1H_mv_va + _externalGrid_Bus_3_p_TM1H_mv_val + _externalGrid_Bus_4_p_TM1H_mv_val)
- (_roof_1_Bus_4_p_TM1H_mv_val + _roof_2_Bus_4_p_TM1H_mv_val + _roof_Bus_1_p_TM1H_mv_val)
+ (_Accumulator_Bus_1_p_TM1H_mv_val + _Accumulator_Bus_2_p_TM1H_mv_val + _Accumulator_Bus_3_p_TM1H_mv_val + _Accumulator_Bus_4_p_TM1H_mv_val)
```

Датасет building_data_meters содержит измерения коммерческого учета
электроэнергии суммарных нагрузок по вводам (1,2,3,4), без разделения на
управляемую или нет.

Использовать данные из этого датасета или нет - на ваше усмотрение,
однако обращу внимание, что иногда данные в датасете building_data
являются недостоверными или отсутствуют, при этом в датасете
building_data_meters за то же время данные есть. Также в
building_data_meters есть данные повышенной дискретности - 3 минуты.

Мы будем оценивать качество прогноза именно **суммы** всех неуправляемых
нагрузок, однако в датасете есть значения индвидуально по вводам и по
фазам вводов, возможно это поможет при прогнозировании.

Для пункта 3) задания необходимо формировать почасовой прогноз на
следующий день. Т.е. каждый день ровно в 00:00 необходимо определить 24
числа - значения потребления на следующий день. При этом есть
информация о потреблении за все предыдущие дни, но очевидно нет
информации из будущего.

Метрику определим, как среднее RMSE за все **дневные** часы в прогнозном
интервале (т.е. с 7:00 до 22:00 МСК). Прогноз ночного потребления нам
неинтересен. Обратите внимание, в датасете время в UTC.

## Решение

**Данные и ML модель**

Использовались:
- два исходных датасета (данные с датчиков ICP DAS и данные приборов учета)
- два рекомендованных (погода ВДНХ и погода Шереметьево) – температура, давление, облач-сть, влажность
- данные с официальной статистикой по зарегистрированным случаям COVID в Москве
- данные по индексу самоизоляции от Яндекс
- добавлены праздники, выходные
- добавлены время локдауна – задается «вручную» + пользователь может вводить новые даты
- добавлены номера недель и дней в году

В результате обработки данных:
- произведена очистка
- отсеяны лишние признаки (линейно зависящие от целевого параметра, категориальные и т.п.)
- отобраны признаки, имеющиее наилучшую корреляцию с целевым параметром
- добавлены лаги

В результате поиска подходящей модели:
- использована кросс-верификация timeSeriesSplit
- проверено 7 моделей по стандартным метрикам (RMSE, MAPE): LinearRegression, KNeighborsRegressor, XGBRegressor, GradientBoostingRegressor, RandomForestRegressor, SVR, MLPRegressor
- определены 2 лучшие подели XGBRegressor, GradientBoostingRegressor, для которых подобраны лучшие параметры с помощью GridSearch

Результаты прогнозирования:
- Метрика оценивалась по прогнозируемым данным за световой день
- Оценено влияние признаков
- Создан MVP интерфейс по заданию, с возможностью использования доп. датасетов, добавления произвольных дат локдауна: на входе датасеты + вводимые пользователем параметры (локдаун),на выходе – метрика и график проноза

Анализ и разработка:
- Стэк: pandas, numpy, sklearn, holidays – библиотека вых дней, streamlit, docker
- Интерфейс: 0) Анализ в Jupyter Notebook; 1) По условию задачи - две функции – fit (обучение модели) и predict (предсказание на основе обучченной модели); 2) Веб-интерфейс с визулизацией результата предсказания (на streamlit)

Справка по работе скриптов:
**1-ый вариант – интерфейс по функциям из задания**
- Две функции – fit и predict, по умолчанию данные - в папке Data в корне
- fit.py – обучает модель, принимая на вход ДС + параметры от пользователя
- preparation и processing – отдельно вызывать не надо, считывают, сливают и очищают данные для дальнейшей работы
- затем в fit берется два регрессора (XGR и GBR), выбираются лучшие параметры GridSearch, сохраняет обучнную модель в pickle
- main – вспомогательный отладочный скрип для проверки
- predict.py – считывает исторические данные для предсказания + новые доступные, чтобы сделать сдвиг на 1 день назад
- Предсказывает по двум лучшим моделям, потом результат суммируется и делится пополам для наилучшего результата по RMSE
- Выводит дату и значение – по требованиям задачи

**2-ый вариант – веб-интерфейс в streamlit**
- Устанавливаем библиотеку, заметем «streamlit run #путь_к_файлу»
- Загружаем данные по очереди в подписанные поля на открывшейся странице браузера, включая доп данные
- Две функции – fit и predict в app_fit и app_predict соотв.
- fit.py – обучает модель, принимая на вход ДС + параметры от пользователя
- preparation и processing – отдельно вызывать не надо, считывают, сливают и очищают данные для дальнейшей работы
- затем в fit берется два регрессора (XGR и GBR), выбираются лучшие параметры GridSearch’ем, сохраняет в pickle
- predict.py – считывает исторические данные для предсказания + новые доступные, чтобы сделать сдвиг на 1 день назад
- Предсказывает по двум лучшим моделям, потом результат суммируется и делится пополам для наилучшего результата по RMSE
- Выводит график или таблицу по выбору пользоватля
