import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import streamlit as st
from mftool import Mftool
import time
from pmdarima.model_selection import train_test_split
from pmdarima import metrics

st.set_page_config(
    page_title="NAV Prediction App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

mf = Mftool()
data1 = mf.get_scheme_codes()
codes = pd.DataFrame(data1.keys())
codes['MF'] = pd.DataFrame(data1.values())

st.title("Mutual Fund's NAV Prediction")
MF = st.selectbox("Select a Mutual Fund", data1.values())


def get_key_from_values(data, val):
    keys = [k for k, v in data.items() if v == val]
    if keys:
        return keys[0]


key = get_key_from_values(data1, MF)
data = mf.get_scheme_historical_nav(key)
navs = pd.DataFrame(data['data'])
navs['date'] = pd.to_datetime(navs['date'], format='%d-%m-%Y')
navs['nav'] = navs['nav'].astype(float)
navs = navs.loc[::-1].reset_index(drop=True)

x_train, x_test = train_test_split(navs['nav'], test_size=0.2)

data_copy = pd.DataFrame(data['data'])
data_copy['date'] = pd.to_datetime(data_copy['date'], format='%d-%m-%Y')
data_copy['nav'] = data_copy['nav'].astype(float)
data_copy = data_copy[::-1].reset_index(drop=True)

detail = mf.get_scheme_details(key)
d = pd.DataFrame(detail)
d['start_nav'] = detail['scheme_start_date']['nav']
d.drop('nav', axis=0, inplace=True)
d.to_string(index=False)
st.header("Scheme Details")
st.table(d)

with st.spinner('Wait for it...'):
    time.sleep(10)
st.success('Done!')

model = pm.auto_arima(x_train)

# for testing data
n_periods_test = len(x_test)
fc_test = model.predict(n_periods=n_periods_test)

navs['Pred'] = fc_test

model1 = pm.auto_arima(navs['nav'])
n_periods = 20
fc, confint = model1.predict(n_periods=n_periods, return_conf_int=True)

# make series for plotting purpose
index_of_fc = np.arange(len(navs.values), len(navs.values) + n_periods)
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

time = pd.date_range(navs['date'].iloc[-1], periods=20, inclusive='right')
dates = pd.DataFrame()
dates['date'] = pd.DataFrame(time)
dates['nav']=pd.DataFrame(fc_series.values.reshape(-1,1))
dates['date'] = pd.to_datetime(dates['date'], format='%d-%m-%Y')
dates['lower_limit'] = pd.DataFrame(lower_series.values.reshape(-1,1))
dates['upper_limit'] = pd.DataFrame(upper_series.values.reshape(-1,1))

st.header("Accuracy")
value = metrics.smape(x_test,fc_test)
st.subheader(value)

st.text("The model's forcast is acceptable if the accuracy score is below 25")

# Plot
col1, col2 = st.columns(2)
with col1:
    fig = plt.figure(figsize=(12, 6))
    plt.plot(navs['date'], navs['nav'])
    plt.plot(navs['date'],navs['Pred'],color='red')
    plt.plot(dates['date'], dates['nav'], color='darkgreen')
    plt.title("Mutual Fund's nav over time")
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.legend(["Actual","Test Data","Predicted Values"],loc="upper left")
    plt.fill_between(dates['date'],
                 dates['lower_limit'],
                 dates['upper_limit'],
                 color='k', alpha=.15)
    st.pyplot(fig)
with col2:
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(dates['date'], dates['nav'], color='darkgreen')
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.title('Predicted Values Along With Confidence Interval')
    plt.fill_between(dates['date'],
                 dates['lower_limit'],
                 dates['upper_limit'],
                 color='k', alpha=.15)
    st.pyplot(fig1)


st.header("Predicted Values")
st.table(dates)

