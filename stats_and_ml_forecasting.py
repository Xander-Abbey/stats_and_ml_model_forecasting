# @title Default title text
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, InterpolationWarning

import pandas as pd
import matplotlib.pyplot as plt

import requests as rq

import numpy as np

from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.correlation import plot_corr
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

import pmdarima as pm
from pmdarima import StepwiseContext
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import prophet
from prophet import Prophet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tools.eval_measures import rmse, rmspe

import pickle

#--------------------------------------------------------------Warnings--------------------------------------------------------------#

#Silencing warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', InterpolationWarning)

#--------------------------------------------------------------Optimisation--------------------------------------------------------------#

#This function is useful if you need to reduce the memory usage of the time series, though the memory usage reduction may vary,
#and may not necessarily make a large difference in performance if overall available system memory is low.
#Function adapted from [1]
def Reduce_Mem_Usage(df, verbose=True):

    """
    Reduces memory usage of large data sets. May be useful if available system memory is low due to extremely large data sets.
    """

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

#--------------------------------------------------------------Exploratory data analysis and preparation--------------------------------------------------------------#

def Normalise_Data(df: pd.DataFrame):

    """
    Returns a normalised representation of the input data
    """

    #Neither irradiation nor wind speed values can be negative, but this is used in case values in the dataset were erroneously made negative
    maxval = abs(df.max())
    df_new = df / maxval
    #print(maxval)
    return df_new

#Function adapted from [2]
def Z_Score_Outliers(df: pd.DataFrame):

    """
    Determines outliers which lay outside of Z-score threshold from the Z-score statistical method
    """

    threshold = 3
    mean = df.mean()
    std_dev = np.std(df)

    outliers = []

    for data in df:
        z_score = (data - mean) / std_dev
        if (np.abs(z_score) > threshold):
            outliers.append(data)

    #print(f"Number of Z Score outliers: {np.size(outliers)}")
    #print(f"Outliers: {outliers}")

    #plt.plot(outliers)
    #plt.show()

    return outliers

def Outlier_Interpolation_Correction(time_series: pd.DataFrame, title_var: str, show: bool = False):

    """
    Finds and replaces outlier values with interpolated values, using the "pchip" interpolation method

    Args:

    "time_series" pertains to the dataset containing the time series

    "title_var" is the time series quanity to correct

    "show" determines whether or not to display the uncorrected and outlier corrected time
    ""

    Returns:

    Outlier-corrected dataset
    """

    outliers = np.unique(Z_Score_Outliers(time_series))

    fig, (ax1, ax2) = plt.subplots(nrows= 2, ncols= 1, sharex= True, figsize= (16, 9))
    fig.suptitle(f"Uncorrected vs. corrected, normalised data values - {title_var}")
    plt.xlabel("Time")
    plt.xticks(rotation= 45)

    ax1.plot(Normalise_Data(time_series),
             color = "blue",
             label= "Uncorrected")
    ax1.legend(framealpha= 0.2, loc= "upper left")

    for ele in outliers:
        time_series = time_series.apply(lambda x: np.nan if x == ele else x)

    time_series = time_series.interpolate(method= "pchip")

    ax2.plot(Normalise_Data(time_series),
             color = "purple",
             label= "Interpolation-corrected")
    ax2.legend(framealpha= 0.2, loc= "upper left")

    if show == True:
        plt.show()

    else:
        pass

    return time_series

def Variable_Correlation(time_series: pd.DataFrame):

    """
    Determine correlation of variables in the dataset, and displays the correlation as a heatmap.
    """

    col_list = ["G(i)", "H_sun", "T2m", "WS10m"]

    corr= time_series.corr()
    #print(corr)
    plot_corr(dcorr= corr, title= "Time series variable correlation matrix", xnames= col_list, ynames= col_list)
    plt.show()

#--------------------------------------------------------------Data reading, cleaning, and displaying--------------------------------------------------------------#

#Reading and loading historic data file into pandas dataframe
def TimeSeries_PVGIS(lat: float, long: float,
                    tracking_type: int = 0, panel_tilt: float = 0.0, panel_azi: float = 0.0,
                    use_opt_inc: int = 0, use_opt_angles: int = 0):

    """
    Retrieve hourly time series data from PVGIS databases for a given latitude and longitude.

    Args:

    "tracking_type" entails the sun-tracking capabilities of the panel(s), is of integer data type, and corresponds to the following tracking types:
        0=fixed,
        1=single horizontal axis (aligned north-south),
        2=two-axis tracking,
        3=vertical axis tracking,
        4=single horizontal axis (aligned east-west),
        5=single inclined axis (aligned north-south)

    "use_opt_inc" entails whether or not to use the calculated optimal panel inclination angle, is of integer data type, and corresponds to the following options:
        0=False
        1=True

    "use_opt_angles" entails whether or not to use the calculated optimal panel inclination and azimuth angles, is of integer data type and corresponds to the following options:
        0=False
        1=True

    Returns:

    time_series variable
    """

    #PVGIS web API automatically handles parameter conflicts by choosing parameters that take precedence
    # - there is no need to worry about catching user inputs with if-elif-else context managers

    url = f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={lat}&lon={long}&startyear=2005&endyear=2020&trackingtype={tracking_type}&optimalinclination={use_opt_inc}&optimalangles={use_opt_angles}&angle={panel_tilt}&aspect={panel_azi}&components=0&outputformat=csv&browser=1"
    response = rq.get(url= url, timeout= 60)

    file_name = f"Timeseries_{lat}_{long}_trackingtype{tracking_type}_{panel_tilt}deg_{panel_azi}deg_2005_2020.csv"
    open(file_name, "wb").write(response.content)

    time_series = pd.read_csv(filepath_or_buffer= file_name,
                              skiprows=8,
                              skipfooter= 10,
                              engine= "python")

    #Cleaning up the data - converting "time" column to datetimes, and dropping unneccessary columns
    dt_format = "%Y%m%d:%H%M"
    time_series["time"] = pd.to_datetime(arg= time_series["time"],
                                         format= dt_format,
                                         utc= True)

    time_series = time_series.drop(columns= ["Int"])
    time_series.index = time_series["time"]

    #Used to prevent 0-values in the time series. 0-values will make MAPE and RMSPE error metrics incalculable,
    #so these values are replaced with values near 0, which is generally more accurate, as it is unlikely that any of these quantities will
    #will stay AT 0 for any length of time, rather hovering near 0, but are recorded as 0 due to measurement limitations.

    #time_series = time_series.replace(0, 0.01)

    return time_series

#Displaying time series data with matplotlib/pyplot
def Display_TimeSeries_PVGIS(time_series: pd.DataFrame):

    """
    Graphs the time series data obtained from PVGIS.
    """

    fig, (ax1, ax2, ax3) = plt.subplots(nrows= 3, ncols= 1, sharex= True, figsize= (16, 9))
    fig.suptitle("Normalised data values")
    plt.xlabel("Time")
    plt.xticks(rotation= 90)

    ax1.plot(Normalise_Data(time_series["G(i)"]),
             color= "orange",
             label= "Global tilted irradiance")
    ax1.legend(framealpha= 0.2, loc= "upper left")

    ax2.plot(Normalise_Data(time_series["T2m"]),
             color= "red",
             label= "Temperature 2m above ground level")
    ax2.legend(framealpha= 0.2, loc= "upper left")

    ax3.plot(Normalise_Data(time_series["WS10m"]),
             color= "blue",
             label= "Wind speed 10m above ground level")
    ax3.legend(framealpha= 0.2, loc= "upper left")

    plt.tight_layout()
    plt.show()

#Setting the year for which the training data ends, and the testing data start
def Train_Test(time_series: pd.DataFrame, split_date: str):

    """
    Args:

    "time_series" is the time series to be split into training and testing sets.

    "split_date" is the date by which the training and testing sets are split.

    Returns:

    training_set and test_set variables as a tuple. training_set does not include the split date in its set, while test_set does
    """

    #Splitting the data into training and testing data
    #Ensure that the data series you're performing the train/test splitting on has its index as the "time" DateTimeIndex, otherwise you will run into errors
    #Otherwise alter this code to look for the specific column that holds the datetimes
    training_set = time_series.loc[time_series.index < split_date]
    test_set = time_series.loc[time_series.index >= split_date]

    #print(training_set)
    #print(test_set)
    return training_set, test_set

#--------------------------------------------------------------Forecasting with statistical models--------------------------------------------------------------#

#Use this to determine the differencing parameter "d" of the SARIMAX model.
#Function adapted from [3]
def Return_ADF_KPSS(time_series, max_d):

    """
    Build dataframe with ADF statistics and p-value for time series after applying difference on time series

    Args:

    "time_series": Dataframe of univariate time series
    "max_d": Max value of how many times to apply differencing

    Returns:

    Dataframe showing values of ADF statistics and p when applying ADF test after applying d times differencing on a time-series.
    """

    results=[]

    for idx in range(max_d):
        adf_result = adfuller(time_series, autolag='AIC')
        kpss_result = kpss(time_series, regression='c', nlags="auto")
        time_series = time_series.diff().dropna()
        if adf_result[1] <=0.05:
            adf_stationary = True
        else:
            adf_stationary = False
        if kpss_result[1] <=0.05:
            kpss_stationary = False
        else:
            kpss_stationary = True

        #Both stationarity tests are required to agree for the time series to be considered stationary
        stationary = adf_stationary & kpss_stationary

        results.append((idx, adf_result[1], kpss_result[1], adf_stationary, kpss_stationary, stationary))

    #Construct DataFrame
    results_df = pd.DataFrame(results, columns=['d','adf_stats','p-value', 'is_adf_stationary','is_kpss_stationary','is_stationary'])

    #Note that the result may take time to be returned, considering the size of the dataset
    print(results_df)
    return results_df

#Use auto-arima function to automatically search for optimal hyperparameters
#Function adapted from [4]
def Auto_SARIMA(d: int,
                training_set: pd.DataFrame, test_set: pd.DataFrame,
                data_name: str, data_units: str,
                forecast_n_hours_ahead: int,
                include_conf_int: bool = False):

    """
    Generates summary of SARIMA models based on input data, sorted by lowest AIC score, and forecasts based on input data.

    Args:
    "d" is the number of times differencing must be applied on the data set to achieve stationarity.

    "training_set" is the data set to train the model on.

    "testing_set" is the data to compare the model forecast with.

    "data_name" is the title of the data set (e.g. wind speed, wind speed 10m above ground, WS10m, etc), and is used for the forecast plot.

    "data_units is the scientific unit of the data (e.g. m/s)"

    "include_conf_int" determines whether or not to show a 95% confidence interval about the forecast data

    Returns:

    Model summary, plot of the forecast, and returns tuple containing test and future forecast dataset, MAE, MAPE, RMSE, and RMSPE of the model.
    """
    StepwiseContext(max_steps= 4)
    model = pm.auto_arima(y= training_set,
                          seasonal= True, #The models are seasonal
                          m= 24, #Number of periods in a cycle/season
                          d= d, #Differencing parameter
                          D= 1, #Seasonal differencing parameter
                          max_p= 4, #Maximum p values to search for
                          max_q= 4, #Maximum q values to search for
                          max_P= 4, #Maximum P values to search for
                          max_Q= 4, #Maximum Q values to search for
                          information_criterion= "aic", #Judge performance of the model by the AIC metric
                          trace= True,
                          error_action= "ignore",
                          stepwise= True,
                          suppress_warnings= True,
                          maxiter= 2)

    print(model.summary())

    n1 = test_set.shape[0]
    test_forecast = model.predict(n_periods= n1, return_conf_int= True)

    n2 = n1 + forecast_n_hours_ahead
    future_forecast = model.predict(n_periods= n2, return_conf_int= True)

    mae_var = mean_absolute_error(y_true= test_set, y_pred= test_forecast[0])
    mape_var = mean_absolute_percentage_error(y_true= test_set, y_pred= test_forecast[0])
    rmse_var = rmse(x1= test_set, x2= test_forecast[0])
    rmspe_var = rmspe(y= test_set, y_hat= test_forecast[0], zeros= 0.01)

    fig, ax = plt.subplots(nrows= 1, ncols= 1, figsize= (16, 9))
    fig.suptitle(f"SARIMA forecast for {data_name}")
    fig.align_labels()

    plt.xlabel("Time")
    plt.xticks(rotation= 90)

    plt.ylabel(data_units)

    ax.plot(training_set,
            color= "blue",
            label= "Training set")

    ax.plot(test_set,
            color= "orange",
            label= "Test set")

    if include_conf_int == True:
        ax.plot(test_forecast[1][len(test_set)-1:],
                color= "red",
                label= "SARIMA test forecast with 95% confidence interval")

        ax.plot(future_forecast[1],
                color= "brown",
                label= "SARIMA future forecast with 95% confidence interval")

    else:
        ax.plot(test_forecast[0],
                color= "red",
                label= "SARIMA test forecast")

        ax.plot(future_forecast[0][len(test_set)-1:],
                color= "brown",
                label= "SARIMA future forecast")

    #Writing "future_forecast[0][len(test_set)-1:]" ensures that only the future part of the forecast is shown,
    #and connects to the last forecasted value on the graph. If this modifier is not addded,
    #the future_forecast plot will otherwise overlay the test forecast and prevent it from being seen

    plt.legend()
    plt.show()

    print(f"Error metrics for SARIMA test forecast of {data_name}")
    print(f"MAE: {round(mae_var, 3)}")
    print(f"MAPE: {round(mape_var, 3)}")
    print(f"RMSE: {round(rmse_var, 3)}")
    print(f"RMSPE: {round(rmspe_var, 3)}")

    return test_forecast, future_forecast, mape_var, rmspe_var

#Function adapted from [4]
def Holt_Winters_ES(training_set: pd.DataFrame, test_set: pd.DataFrame,
                    data_name: str, data_units: str,
                    forecast_n_hours_ahead: int = 0):

    """
    Generates "Holt-Winters Exponential Smoothing forecast based on input data.

    Args:

    "training_set" is the data to train the model on.

    "testing_set" is the data to compare the model forecast with.

    "data_name" is the title of the data set (e.g. wind speed, wind speed 10m above ground, WS10m, etc), and is used for the forecast plot.

    "data_units is the scientific unit of the data (e.g. m/s)"

    "forecast_n_hours ahead" is the number of hours to forecast for after the end of the test data set.

    Returns:

    Plot of the forecast, and returns tuple containing test and future forecast dataset, MAE, MAPE, RMSE, and RMSPE of the model.
    """

    model = ExponentialSmoothing(endog= training_set,
                                 trend= "add",
                                 seasonal= "add",
                                 seasonal_periods= 24,
                                 damped_trend= False).fit(optimized= True,
                                                          remove_bias= False)

    #The -1 is there due to the overlap between the last date in the training set and the first day in the test set
    s1 = training_set.shape[0]
    e1 = s1 + test_set.shape[0] - 1
    test_forecast = model.predict(start= s1, end= e1)

    s2 = training_set.shape[0] + test_set.shape[0] - 1
    e2 = s2 + forecast_n_hours_ahead
    future_forecast = model.predict(start= s2, end= e2)

    mae_var = mean_absolute_error(y_true= test_set, y_pred= test_forecast)
    mape_var = mean_absolute_percentage_error(y_true= test_set, y_pred= test_forecast)
    rmse_var = rmse(x1= test_set, x2= test_forecast)
    rmspe_var = rmspe(y= test_set, y_hat= test_forecast, zeros= 0.01)

    fig, ax = plt.subplots(nrows= 1, ncols= 1, figsize= (16, 9))
    fig.suptitle(f"Holt-Winters ETS forecast for {data_name}")
    fig.align_labels()

    plt.xlabel("Time")
    plt.xticks(rotation= 90)

    plt.ylabel(data_units)

    ax.plot(training_set,
            color= "blue",
            label= "Training set")

    ax.plot(test_set,
            color= "orange",
            label= "Test set")

    ax.plot(test_forecast,
            color= "red",
            label= "Holt-Winters Exponential Smoothing test forecast")

    if forecast_n_hours_ahead > 0:
      ax.plot(future_forecast,
              color= "brown",
              label= "Holt-Winters Exponential Smoothing future forecast")

    else:
      pass

    plt.legend(loc= "upper left")
    plt.show()

    print(f"Error metrics for Holt-Winters Exponential Smoothing test forecast of {data_name}")
    print(f"MAE: {round(mae_var, 3)}")
    print(f"MAPE: {round(mape_var, 3)}")
    print(f"RMSE: {round(rmse_var, 3)}")
    print(f"RMSPE: {round(rmspe_var, 3)}")

    return test_forecast, future_forecast, mape_var, rmspe_var

#Robust statistical model created by Facebook's Core Data Science team - highly suited to modelling and forecasting data with strong seasonality
#Function adapted from [5][6]
def FB_Prophet(time_series: pd.DataFrame,
               split_date: str,
               var_to_forecast: str, data_name: str, data_units: str,
               growth: str, forecast_n_hours_ahead: int):

    """
    Generates a forecast from the Facebook Prophet additive regression model based on input data.

    Args:

    "time_series" is the full data set obtained from PVGIS.

    "split_date" is the date by which the training and testing sets are split.

    "var_to_forecast" refers to the time series quantity being modelled.

    "data_name" is the title of the data set (e.g. wind speed, wind speed 10m above ground, WS10m, etc), and is used for the forecast plot.

    "data_units is the scientific unit of the data (e.g. m/s)"

    "growth" refers to the nature of the time series trend component. Valid options are either "flat" or linear"

    "forecast_n_hours_ahead" is how many hours to forecast for, after the last datetime in the test set.

    Returns:

    Plot of the forecast, and tuple containing test and future forecast datasets, MAE, MAPE, RMSE, and RMSPE of the model.
    """

    time_series["time"] = time_series["time"].dt.tz_localize(None)

    training_set, test_set = Train_Test(time_series, split_date)
    training_set = training_set.reset_index(drop= True)
    training_set = training_set.rename(columns= {"time": "ds", var_to_forecast: "y"})
    training_set = training_set.filter(items= ["ds", "y"])

    test_set = test_set.reset_index(drop= True)
    test_set = test_set.rename(columns= {"time": "ds", var_to_forecast: "y"})
    test_set = test_set.filter(items= ["ds", "y"])

    #Have a closer look at what data is being used for testing and training
    #and how you manage the future_datetimes dataframe creation
    model = Prophet(growth = growth, interval_width= 0, daily_seasonality= True)
    model = model.fit(df= training_set)

    p1 = len(test_set)
    test_datetimes = model.make_future_dataframe(periods= p1,
                                                 freq= "H",
                                                 include_history= False)

    #The +1 is because the last and first data points in the test_datetimes and future_datetimes overlap, so an extra datapoint is needed
    #to ensure the full data points for forecast_n_hours_ahead are made
    p2 = p1 + forecast_n_hours_ahead + 1
    future_datetimes = model.make_future_dataframe(periods= p2,
                                                   freq= "H",
                                                   include_history= False)

    test_forecast = model.predict(df= test_datetimes)
    future_forecast = model.predict(df= future_datetimes)

    mae_var = mean_absolute_error(y_true= test_set["y"], y_pred= test_forecast["yhat"])
    mape_var = mean_absolute_percentage_error(y_true= test_set["y"], y_pred= test_forecast["yhat"])
    rmse_var = rmse(x1= test_set["y"], x2= test_forecast["yhat"])
    rmspe_var = rmspe(y= test_set["y"], y_hat= test_forecast["yhat"])

    fig, ax = plt.subplots(nrows= 1, ncols= 1, figsize= (16, 9))
    fig.suptitle(f"Prophet forecast for {data_name}")
    fig.align_labels()

    plt.xlabel("Time")
    plt.xticks(rotation= 90)

    plt.ylabel(data_units)

    #Specifying the datetimes for the x-axis, and the y/yhat values for the y-axis is necessary, as the pyplot seems to
    #display the data incorrectly otherwise, and mixing the prophet plotter and pyplot produces similarly disastrous display results
    ax.plot(training_set["ds"], training_set["y"],
            color= "blue",
            label= "Training set")

    ax.plot(test_set["ds"], test_set["y"],
            color= "orange",
            label= "Test set")

    ax.plot(test_datetimes, test_forecast["yhat"],
           color= "red",
           label= "Prophet test forecast")

    #The term [len(test_forecast["yhat"])-1:] is added on to ensure the future_forecast doesn't overlap with the test_forecast
    #as all forecasts are made after the training set, so overlapping must be accounted for.
    #The -1 ensures that the last point in the test forecast and the first point in the future forecast overlap,
    #thus ensuring graph continuity

    ax.plot(future_datetimes[len(test_forecast["yhat"])-1:], future_forecast["yhat"][len(test_forecast["yhat"])-1:],
        color= "brown",
        label= "Prophet future forecast")

    plt.legend(loc= "upper left")
    plt.show()

    print(f"Error metrics for Facebook Prophet test forecast of {data_name}")
    print(f"MAE: {round(mae_var, 3)}")
    print(f"MAPE: {round(mape_var, 3)}")
    print(f"RMSE: {round(rmse_var, 3)}")
    print(f"RMSPE: {round(rmspe_var, 3)}")

    return test_forecast, future_forecast, mae_var, mape_var, rmse_var, rmspe_var

#--------------------------------------------------------------Forecasting with machine learning models--------------------------------------------------------------#

#Function adapted from [6][7]
def XGBoost(ml_time_series: pd.DataFrame, var_to_forecast: str,
            split_date: str, forecast_n_hours_ahead: int,
            data_name: str, data_units: str):

    """
    Args:

    "ml_time_series" refers to either a univariate or multivariate time series, though the outputs are univariate.
    
    "var_to_forecast" refers to the time series quantity being modelled.

    "split_date" is the date by which the training and testing sets are split.

    "forecast_n_hours_ahead" is how many hours to forecast for, after the last datetime in the test set.

    "data_name" is the title of the data set (e.g. wind speed, wind speed 10m above ground, WS10m, etc), and is used for the forecast plot.

    "data_units is the scientific unit of the data (e.g. m/s)"

    "Returns:

    Plot of the forecast, and tuple containing test and future forecast datasets, MAE, MAPE, RMSE, and RMSPE of the model.
    """

    #Decided to reassign variable for ease of typing. Nothing would change if all the forecast_var instances were replaced with var_to_forecast
    forecast_var = var_to_forecast

    #Performing feature engineering to turn datetimes into numerical values for the ML model to train on
    def Feature_Adding(time_series: pd.DataFrame):

        """
        Add necessary time-based features to reframe the problem into a supervised learning problem
        """

        time_series["hourofday"] = time_series["time"].dt.hour
        time_series["dayofyear"] = time_series["time"].dt.dayofyear
        time_series["dayofweek"] = time_series["time"].dt.dayofweek
        time_series["dayofmonth"] = time_series["time"].dt.day
        time_series["month"] = time_series["time"].dt.month
        time_series["quarter"] = time_series["time"].dt.quarter
        time_series["year"] = time_series["time"].dt.year

        #print(time_series)
        #Note that the returned series duplicates the "time"-labelled index, and creates a "time"-labelled column from which the features are created
        #This ensures the DataFrames are easily sorted by DateTimeIndex, and features can be made from the normal time column
        #A different, more efficient approach may be taken, though this worked well enough, though is likely to be changed later
        return time_series

    def Future_DataFrame(time_series: pd.DataFrame):
        last_date = time_series.index.max()
        date_range = pd.date_range(start= last_date, periods= forecast_n_hours_ahead, freq= "H")
        future_df = pd.DataFrame(index= date_range)
        future_df["time"] = date_range
        future_df["is_future"] = True
        time_series["future"] = False
        future_df = Feature_Adding(future_df)

        ts_and_future = pd.concat([time_series, future_df])
        ts_and_future = Feature_Adding(time_series= ts_and_future)

        print("ts and future: ")
        print(ts_and_future)

        future_with_features = ts_and_future[len(time_series):]
        print("future with features: ")
        print(future_with_features)

        return future_with_features

    time_series = Feature_Adding(ml_time_series)
    train, test = Train_Test(time_series, split_date)
    features = ["hourofday", "dayofyear", "dayofweek", "dayofmonth", "month", "quarter", "year"]

    X_train = train[features]
    y_train = Outlier_Interpolation_Correction(train[forecast_var], forecast_var)
    print("X_train: ")
    print(X_train)

    X_test = test[features]
    y_test = Outlier_Interpolation_Correction(test[forecast_var], forecast_var, False)
    print("X_test: ")
    print(X_test)

    model = XGBRegressor(n_estimators= 1000,
                         early_stopping_rounds= 100,
                         learning_rate = 0.1)
    model.fit(X= X_train, y= y_train,
              eval_set= [[X_train, y_train]],
              verbose= 100)

    feat_imp_df = pd.DataFrame(data= model.feature_importances_,
                               index= model.feature_names_in_,
                               columns= ["importance"])
    feat_imp_df.sort_values("importance").plot(kind= "barh", title= f"XGBoost feature importance for {forecast_var}")
    plt.show()

    test["test_forecast"] = model.predict(X_test, validate_features= True)

    future_with_forecast = Future_DataFrame(ml_time_series)
    future_with_forecast["future_forecast"] = model.predict(future_with_forecast[features])
    print("future forecast: ", future_with_forecast["future_forecast"])

    mae_var = mean_absolute_error(y_true= test[forecast_var], y_pred= test["test_forecast"])
    mape_var = mean_absolute_percentage_error(y_true= test[forecast_var], y_pred= test["test_forecast"])
    rmse_var = rmse(x1= test[forecast_var], x2= test["test_forecast"])
    rmspe_var = rmspe(y= test[forecast_var], y_hat= test["test_forecast"], zeros= 0.01)

    time_series = time_series.merge(test[["test_forecast"]],
                                    how= "left",
                                    left_index= True,
                                    right_index= True)

    fig, ax = plt.subplots(figsize= (16, 9))
    train[forecast_var].plot(ax= ax)
    test[forecast_var].plot(ax= ax)
    time_series["test_forecast"].plot(ax= ax)
    future_with_forecast["future_forecast"].plot(ax= ax)
    plt.legend(["Train", "Test", "Test forecast", "Future forecast"], loc= "upper left")
    plt.suptitle(f"XGBoost forecast for {data_name}")
    plt.xlabel("Time")
    plt.xticks(rotation= 90)
    plt.ylabel(data_units)
    plt.show()

    print(f"Error metrics for XGBoost test forecast of {data_name}")
    print(f"MAE: {round(mae_var, 3)}")
    print(f"MAPE: {round(mape_var, 3)}")
    print(f"RMSE: {round(rmse_var, 3)}")
    print(f"RMSPE: {round(rmspe_var, 3)}")

    return model, test["test_forecast"], future_with_forecast["future_forecast"], mae_var, mape_var, rmse_var, rmspe_var

def LGBM(ml_time_series: pd.DataFrame, var_to_forecast: str,
         split_date: str, forecast_n_hours_ahead: int,
         data_name: str, data_units: str):

    """
    Args:

    "ml_time_series" refers to either a univariate or multivariate time series, though the outputs are univariate.
    
    "var_to_forecast" refers to the time series quantity being modelled.

    "split_date" is the date by which the training and testing sets are split.

    "forecast_n_hours_ahead" is how many hours to forecast for, after the last datetime in the test set.

    "data_name" is the title of the data set (e.g. wind speed, wind speed 10m above ground, WS10m, etc), and is used for the forecast plot.

    "data_units is the scientific unit of the data (e.g. m/s)"

    "Returns:

    Plot of the forecast, and tuple containing test and future forecast datasets, MAE, MAPE, RMSE, and RMSPE of the model.
    """

    #Decided to reassign variable for ease of typing. Nothing would change if all the forecast_var instances were replaced with var_to_forecast
    forecast_var = var_to_forecast

    #Performing feature engineering to turn datetimes into numerical values for the ML model to train on
    def Feature_Adding(time_series: pd.DataFrame):

        """
        Add necessary features to reframe the problem into a supervised learning problem
        """

        time_series = time_series.copy()
        time_series["hourofday"] = time_series["time"].dt.hour
        time_series["dayofyear"] = time_series["time"].dt.dayofyear
        time_series["dayofweek"] = time_series["time"].dt.dayofweek
        time_series["dayofmonth"] = time_series["time"].dt.day
        time_series["month"] = time_series["time"].dt.month
        time_series["quarter"] = time_series["time"].dt.quarter
        time_series["year"] = time_series["time"].dt.year
        #print(time_series)

        #Note that the returned series duplicates the "time"-laballed index, and creates a "time"-labelled column from which the features are created
        return time_series

    def Future_DataFrame(time_series: pd.DataFrame):
        last_date = time_series.index.max()
        date_range = pd.date_range(start= last_date, periods= forecast_n_hours_ahead, freq= "H")
        future_df = pd.DataFrame(index= date_range)
        future_df["time"] = date_range
        future_df["is_future"] = True
        time_series["future"] = False
        future_df = Feature_Adding(future_df)

        ts_and_future = pd.concat([time_series, future_df])
        ts_and_future = Feature_Adding(time_series= ts_and_future)

        print("ts and future: ")
        print(ts_and_future)

        future_with_features = ts_and_future[len(time_series):]
        print("future with features: ")
        print(future_with_features)

        return future_with_features

    time_series = Feature_Adding(ml_time_series)
    train, test = Train_Test(time_series, split_date)
    features = ["hourofday", "dayofyear", "dayofweek", "dayofmonth", "month", "quarter", "year"]

    X_train = train[features]
    y_train = Outlier_Interpolation_Correction(train[forecast_var], forecast_var)
    print("X_train: ")
    print(X_train)

    X_test = test[features]
    y_test = Outlier_Interpolation_Correction(test[forecast_var], forecast_var, False)
    print("X_test: ")
    print(X_test)

    model = LGBMRegressor(n_estimators= 1000,
                         early_stopping_rounds= 100,
                         learning_rate = 0.1)
    model.fit(X= X_train, y= y_train,
              eval_set= [[X_train, y_train]])

    feat_imp_df = pd.DataFrame(data= model.feature_importances_,
                               index= model.feature_name_,
                               columns= ["importance"])
    feat_imp_df.sort_values("importance").plot(kind= "barh", title= f"LightGBM feature importance for {forecast_var}")
    plt.show()

    test["test_forecast"] = model.predict(X_test)

    future_with_forecast = Future_DataFrame(ml_time_series)
    future_with_forecast["future_forecast"] = model.predict(future_with_forecast[features])
    print("future forecast: ", future_with_forecast["future_forecast"])

    mae_var = mean_absolute_error(y_true= test[forecast_var], y_pred= test["test_forecast"])
    mape_var = mean_absolute_percentage_error(y_true= test[forecast_var], y_pred= test["test_forecast"])
    rmse_var = rmse(x1= test[forecast_var], x2= test["test_forecast"])
    rmspe_var = rmspe(y= test[forecast_var], y_hat= test["test_forecast"], zeros= 0.01)

    time_series = time_series.merge(test[["test_forecast"]],
                                    how= "left",
                                    left_index= True,
                                    right_index= True)

    fig, ax = plt.subplots(figsize= (16, 9))
    train[forecast_var].plot(ax= ax)
    test[forecast_var].plot(ax= ax)
    time_series["test_forecast"].plot(ax= ax)
    future_with_forecast["future_forecast"].plot(ax= ax)
    plt.legend(["Train", "Test", "Test forecast", "Future forecast"], loc= "upper left")
    plt.suptitle(f"LightGBM forecast for {data_name}")
    plt.xlabel("Time")
    plt.xticks(rotation= 90)
    plt.ylabel(data_units)
    plt.show()

    print(f"Error metrics for LGBM test forecast of {data_name}")
    print(f"MAE: {round(mae_var, 3)}")
    print(f"MAPE: {round(mape_var, 3)}")
    print(f"RMSE: {round(rmse_var, 3)}")
    print(f"RMSPE: {round(rmspe_var, 3)}")

    return model, test["test_forecast"], future_with_forecast["future_forecast"], mae_var, mape_var, rmse_var, rmspe_var

#--------------------------------------------------------------Testing--------------------------------------------------------------#

##Latitude and longitude values correspond to Pretoria, South Africa (25.7479 S, 28.2293 E).

##Solar panel values correspond to a 2-axis tracking system.

##The returned file contains weather data variables indexed by datetime values from 2005-01-01 -- 2020-12-31 for a PVGIS-SARAH2 database -
##those who fall out of this satellite and database coverage need to alter the PVGIS API URL's start and end years to match the available years
##for the databases that cover their area.

#V#isit https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en for details

##Time series retrieval
ts = TimeSeries_PVGIS(-25.7479, 28.2293, 2, 0, 0, 1, 1)

datetime_start = "2020-12-01"
ts = ts[datetime_start:]

ml_datetime_start = "2020-12-01"
ml_ts = ts[ml_datetime_start:]

vc = Variable_Correlation(ts)

display_ts = Display_TimeSeries_PVGIS(ts)

#Look at that Durban/Steve Biko location dataset again as an example of a dataset containing highly anomalous data, illustrating the need for the outlier correction method
corrected_gti_ts = Outlier_Interpolation_Correction(ts["G(i)"], "G(i)")
corrected_t2m_ts = Outlier_Interpolation_Correction(ts["T2m"], "T2m")
corrected_ws10m_ts = Outlier_Interpolation_Correction(ts["WS10m"], "WS10m")

train_set_gti, test_set_gti = Train_Test(corrected_gti_ts, "2020-12-27")
train_set_t2m, test_set_t2m = Train_Test(corrected_t2m_ts, "2020-12-27")
train_set_ws10m, test_set_ws10m = Train_Test(corrected_ws10m_ts, "2020-12-27")

#PACF and ACF plots:
fig1, (ax1, ax2) = plt.subplots(ncols = 2, nrows= 3, figsize= (16, 9))

fig1.align_labels()
plt.xlabel("Lags")

pacf1 = plot_pacf(x= corrected_gti_ts, ax= ax1)
acf1 = plot_acf(x= corrected_gti_ts, ax= ax2)

pacf2 = plot_pacf(x= corrected_t2m_ts, ax= ax3)
acf2 = plot_pacf(x= corrected_t2m_ts, ax= ax4)

pacf3 = plot_pacf(x= corrected_ws10m_ts, ax= ax5)
acf3 = plot_pacf(x= corrected_ws10m_ts, ax= ax6)

plt.show()

#Decomposition plots:
plt.figure(figsize=(16,9))

decomp1 = seasonal_decompose(corrected_gti_ts)
decomp1.plot()
decomp2 = seasonal_decompose(corrected_t2m_ts)
decomp2.plot()
decomp3 = seasonal_decompose(corrected_ws10m_ts)
decomp3.plot()

plt.show()

#ADF and KPSS tests:
d_test1 = Return_ADF_KPSS(corrected_gti_ts, 3)
d_test2 = Return_ADF_KPSS(corrected_t2m_ts, 3)
d_test3 = Return_ADF_KPSS(corrected_ws10m_ts, 3)

#Models:
#Statistical
auto_sarima_gti = Auto_SARIMA(0,
                              train_set_gti, test_set_gti,
                              "global tilted irradiance", "W/m\u00b2",
                              48)

auto_sarima_t2m = Auto_SARIMA(1,
                             train_set_t2m, test_set_t2m,
                             "temperature 2m above ground", "\u00b0" + "C",
                             48)

auto_sarima_ws10m = Auto_SARIMA(1,
                               train_set_ws10m, test_set_ws10m,
                               "wind speed 10m above ground", "m/s",
                               48)

hw_gti = Holt_Winters_ES(train_set_gti, test_set_gti,
                        "global tilted irradiance", "W/m\u00b2",
                        48)

hw_t2m = Holt_Winters_ES(train_set_t2m, test_set_t2m,
                        "temperature 2m above ground", "\u00b0" + "C",
                        48)

hw_ws10m = Holt_Winters_ES(train_set_ws10m, test_set_ws10m,
                        "wind speed 10m above ground", "m/s",
                        48)

prophet_gti = FB_Prophet(ts,
                        "2020-12-27",
                        "G(i)", "global tilted irradiance", "W/m\u00b2",
                        "flat",
                        48)

prophet_t2m = FB_Prophet(ts,
                        "2020-12-27",
                        "T2m", "temperature 2m above ground", "\u00b0" + "C",
                        "flat",
                        48)

prophet_ws10m = FB_Prophet(ts,
                          "2020-12-27",
                          "WS10m", "wind speed 10m above ground", "m/s",
                          "flat",
                          48)

#Machine learning
xgb_gti = XGBoost(ml_ts, "G(i)",
                 "2020-12-27",
                 48,
                 "global tilted irradiance",
                 "W/m\u00b2")

xgb_t2m = XGBoost(ml_ts, "T2m",
                 "2020-12-27",
                 48,
                 "temperature 2m above ground",
                 "\u00b0" + "C")

xgb_ws10m = XGBoost(ml_ts, "WS10m",
                   "2020-12-27",
                   48,
                   "wind speed 10m above ground",
                   "m/s")

lgbm_gti = LGBM(ml_ts, "G(i)",
               "2020-12-27",
               48,
               "global tilted irradiance",
               "W/m\u00b2")

lgbm_t2m = LGBM(ml_ts, "T2m",
               "2020-12-27",
               48,
               "temperature 2m above ground",
               "\u00b0" + "C")

lgbm_ws10m = LGBM(ml_ts, "WS10m",
                 "2020-12-27",
                 48,
                 "wind speed 10m above ground",
                 "m/s")

#--------------------------------------------------------------References#--------------------------------------------------------------#

#[1] https://github.com/MKB-Datalab/time-series-analysis-with-SARIMAX-and-Prophet/blob/master/notebooks/02-Forecasting_with_SARIMAX.ipynb

#[2] https://www.analyticsvidhya.com/blog/2022/10/outliers-detection-using-iqr-z-score-lof-and-dbscan/

#[3] https://www.kaggle.com/code/yuchengkuo/forecasting-and-time-series-analysis-with-python

#[4] https://medium.com/analytics-vidhya/ml22-6318a9c9dc35

#[5] https://facebook.github.io/prophet/docs/quick_start.html

#[6] https://medium.com/mlearning-ai/time-series-forecasting-with-xgboost-and-lightgbm-predicting-energy-consumption-460b675a9cee

#[7] https://www.kaggle.com/code/robikscube/pt2-time-series-forecasting-with-xgboost/notebook#Reviewing:-Train-/-Test-Split

