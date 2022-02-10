import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64

st.title('Forecasting Model')

"""
This data app uses Facebook's open-source Prophet library to automatically generate future forecast values from an imported dataset.
You'll be able to import your data from a CSV file, visualize trends and features, analyze forecast performance, and finally download the created forecast 😵 
"""

"""
### Step 1: Import Data
"""
df = st.file_uploader(
    'Import the time series csv file here. Columns must be labeled ds and y. The input to Prophet is always a '
    'dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, '
    'ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, '
    'and represents the measurement we wish to forecast.',
    type='csv', encoding='auto')

if df is not None:
    data = pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'], errors='coerce')

    st.write(data)

    max_date = data['ds'].max()
    # st.write(max_date)

periods_input = st.number_input('How many periods would you like to forecast into the future?',
                                min_value=1, max_value=365)

if df is not None:
    m = Prophet()
    m.fit(data)

if df is not None:
    future = m.make_future_dataframe(periods=periods_input)

    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered = fcst[fcst['ds'] > max_date]
    st.write(fcst_filtered)

    fig1 = m.plot(forecast)
    st.write(fig1)

    fig2 = m.plot_components(forecast)
    st.write(fig2)

if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** ' \
           f'&lt;forecast_name&gt;.csv**) '
    st.markdown(href, unsafe_allow_html=True)
