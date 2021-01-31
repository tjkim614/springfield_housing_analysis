import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.style.use('seaborn-talk')
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

def arima_cross_validation(data, order, initial=12*15, horizon=12, period=6, verbose=False):
    k = (len(data)-initial-horizon)//period
    if verbose: print('Cross validating over', str(k), 'folds.')
    
    rmses = []
    for i in range(1, k+1):
        n = len(data)-horizon-((k-i)*period)
        model = ARIMA(data[:n], order=order, freq='MS').fit()
        y_hat = model.get_forecast(steps=horizon).predicted_mean.to_numpy()
        y = data[n:n+horizon].to_numpy()
        rmse = np.sqrt(mean_squared_error(y, y_hat))
        if verbose: print(f'fold {i}: train[0:{n}], test[{n}:{n+horizon}] of {len(data)}, rmse={rmse}')
        rmses.append(rmse)

    return rmses

def arima_analyze(data, order, initial=12*15, horizon=12, period=6, forecast_length=24, filename=None):
    
    forecast_index = pd.date_range(data.index[-1], periods=forecast_length+1, freq='MS')[1:]
    forecast_df = pd.DataFrame(index=forecast_index)
    
    n = len(data.columns)
    rows, cols = -(-n//2), (1+(n>1))
    fig = plt.figure(figsize=(10*cols, 6*rows))
    axs = fig.subplots(rows, cols, squeeze=False).flatten()
    
    for i, col in enumerate(data.columns):
        #cross-validation to get rmse
        rmses = arima_cross_validation(data=data[col], order=order, initial=initial, horizon=horizon, period=period)
        rmse = sum(rmses)/len(rmses)
        
        #run model to get 1-year forecast
        model = ARIMA(data[col], order=order, freq='MS').fit()
        prediction_results = model.get_prediction(start=2, typ='levels')
        prediction = prediction_results.predicted_mean
        prediction_conf_int = prediction_results.conf_int(alpha=0.2)
        prediction_lower = prediction_conf_int[f'lower {col}'].tolist()
        prediction_upper = prediction_conf_int[f'upper {col}'].tolist()
        
        forecast_results = model.get_forecast(steps=forecast_length)
        forecast = forecast_results.predicted_mean
        forecast_df[col] = forecast
        forecast_conf_int = forecast_results.conf_int(alpha=0.2)
        forecast_lower = forecast_conf_int[f'lower {col}'].tolist()
        forecast_upper = forecast_conf_int[f'upper {col}'].tolist()
        
        #plot data with forecast
        ax = axs[i]
        ax.plot(data.index, data[col], 'k.')
        ax.plot(data.index[-len(prediction):], prediction, ls='-', c='#0072B2')
        ax.plot(forecast_index, forecast, ls='-', c='#0072B2')
        ax.fill_between(prediction.index, prediction_lower, prediction_upper, color='#0072B2', alpha=0.2)
        ax.fill_between(forecast_index, forecast_lower, forecast_upper, color='#0072B2', alpha=0.2)
        ax.set_title(f'{col}, rmse: {int(rmse)}')
        ax.legend(labels=['actual', 'model'], loc='upper left')
    
    if i>1:
        plt.suptitle('ARIMA Models', y=1.03, fontsize=30)
    plt.tight_layout()
    if filename:
        plt.savefig(f'visualizations/{filename}.png')
    plt.show()
    
    return forecast_df

def acf_pacf_charts(df, filename=None):
    n = len(df.columns)
    fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(12, 4*n))
    fig.suptitle('Autocorrelations by Zip Code', y=1.03, fontsize=30)

    for i, col in enumerate(df.columns):
        plot_acf(df[col], ax=axs[i][0], title='ACF: '+col)
        plot_pacf(df[col], ax=axs[i][1], title='PACF: '+col)

    plt.tight_layout()
    #fig.subplots_adjust(hspace=.41, wspace=.17)
    if filename:
        plt.savefig(f'visualizations/{filename}.png')
    plt.show()

#obsolete: use arima_analyze or arima_cross_validation
def plot_summary_forecast(arima_mod, test_df, df_col):
    print(arima_mod.summary())
    # forecast
    arima_forecast = arima_mod.forecast(steps=len(test_df))[0]
    
    # plotting forecast vs actual
    plt.figure(figsize=(20,10))

    forecast_df = pd.DataFrame(arima_forecast, test_df.index)

    plt.plot(test_df[df_col])
    plt.plot(forecast_df)

    plt.legend(('Data', 'Predictions'), fontsize=16)
    plt.title(df_col + ' : Difference in House Sales')
    plt.ylabel('Sales', fontsize=16);
    
    plt.show()

    #RMSE
    RMSE = np.sqrt(mean_squared_error(test_df[df_col], arima_forecast))
    print('RMSE:', round(RMSE), '\n \n \n')