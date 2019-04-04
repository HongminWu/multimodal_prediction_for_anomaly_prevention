import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import download_plotlyjs, plot
import numpy as np
import pandas as pd
import ipdb

def plot_anomaly_with_plotly(df, anomaly_t_by_human = None, metric_name=None, filename = 'default'):
    dates = df.load_date
    bool_array = (abs(df['anomaly_points']) > 0)
    #And a subplot of the Actual Values.
    actuals = df["actuals"][-len(bool_array):]
    anomaly_points = bool_array * actuals
    anomaly_points[anomaly_points == 0] = np.nan
    color_map= {0: "'rgba(228, 222, 249, 0.65)'", 1: "yellow", 2: "orange", 3: "red"}

    table = go.Table(
                        domain=dict(x=[0, 1],
                                    y=[0, 0.3]),
                        columnwidth=[1, 2 ],
                        header = dict(height = 20,
                                      values = [['<b>Date</b>'],
                                                ['<b>Actual Values </b>'],
                                                ['<b>Predicted</b>'],
                                                ['<b>% Difference</b>'],
                                                ['<b>Severity (0-3)</b>']],
                                      font  = dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                                      fill  = dict(color='#d562be')),
                                      cells = dict(
                                             values = [df.round(3)[k].tolist() for k in ['load_date', 'actuals', 'predicted',
                                                                                       'percentage_change','color']],
                                             line = dict(color='#506784'),
                                             align = ['center'] * 5,
                                             font = dict(color=['rgb(40, 40, 40)'] * 5, size=12),
                                             suffix=[None] + [''] + [''] + ['%'] + [''],
                                             height = 27,
                                             fill=dict(color='rgb(245,245,245)'), #unique color for the first column [df['color'].map(color_map)],)
                                             )
                    )
    
    anomalies = go.Scatter(name="Anomaly",
                           x=dates,
                           xaxis='x1',
                           yaxis='y1',
                           y=df['anomaly_points'],
                           mode='markers',
                           marker = dict(color ='red',
                               size = 11,
                               line = dict(
                                         color = "red",
                                         width = 2))
                           )
    
    upper_bound = go.Scatter(hoverinfo="skip",
                         x=dates,
                         showlegend =False,
                         xaxis='x1',
                         yaxis='y1',
                         y=df['3s'],
                         marker=dict(color="#444"),
                         line=dict(
                             color=('rgb(23, 96, 167)'),
                             width=2,
                             dash='dash'),
                         fillcolor='rgba(68, 68, 68, 0.3)',
                         fill='tonexty')
    
    lower_bound = go.Scatter(name='Confidence Interval',
                          x=dates,
                         xaxis='x1',
                         yaxis='y1',
                          y=df['-3s'],
                          marker=dict(color="#444"),
                          line=dict(
                              color=('rgb(23, 96, 167)'),
                              width=2,
                              dash='dash'),
                          fillcolor='rgba(68, 68, 68, 0.3)',
                          fill='tonexty')
    
    Actuals = go.Scatter(name= 'Actuals',
                     x= dates,
                     y= df['actuals'],
                     xaxis='x2',
                     yaxis='y2',
                     mode='lines',
                     marker=dict(size=12,
                                 line=dict(width=1),
                                 color="blue"))

    Predicted = go.Scatter(name= 'Predicted',
                     x= dates,
                     y= df['predicted'],
                    xaxis='x2', yaxis='y2',
                     mode='lines',
                     marker=dict(size=12,
                                 line=dict(width=1),
                                 color="orange"))
    # create plot for error...
    Error = go.Scatter(name="Error",
                   x=dates, y=df['error'],
                   xaxis='x1',
                   yaxis='y1',
                   mode='lines',
                   marker=dict(size=12,
                               line=dict(width=1),
                               color="red"),
                   text="Error")
    
    anomalies_map = go.Scatter(name = "anomaly actual",
                                   showlegend=False,
                                   x=dates,
                                   y=anomaly_points,
                                   mode='markers',
                                   xaxis='x2',
                                   yaxis='y2',
                                    marker = dict(color ="red",
                                  size = 11,
                                 line = dict(
                                     color = "red",
                                     width = 2)))
    
    Mvingavrg = go.Scatter(name="Moving Average",
                           x=dates,
                           y=df['meanval'],
                           mode='lines',
                           xaxis='x1',
                           yaxis='y1',
                           marker=dict(size=12,
                                       line=dict(width=1),
                                       color="green"),
                           text="Moving average")

    axis=dict(
        showline=True,
        zeroline=False,
        showgrid=True,
        mirror=True,
        ticklen=4,
        gridcolor='#ffffff',
        tickfont=dict(size=10))
    
    layout = dict(
        width=1800,
        height=865,
        autosize=False,
        title= metric_name,
        margin = dict(t=75),
        showlegend=True,
        xaxis1=dict(axis, **dict(domain=[0, 1], anchor='y1', showticklabels=True)),
        xaxis2=dict(axis, **dict(domain=[0, 1], anchor='y2', showticklabels=True)),
        yaxis1=dict(axis, **dict(domain=[2 * 0.21 + 0.20 + 0.09, 1], anchor='x1', hoverformat='.2f')),
        yaxis2=dict(axis, **dict(domain=[0.21 + 0.12, 2 * 0.31 + 0.02], anchor='x2', hoverformat='.2f')))
    

    fig = go.Figure(data = [table,
            anomalies,
            anomalies_map,
            upper_bound,
            lower_bound,
            Actuals,
            Predicted,
            Mvingavrg,
            Error,
            ],
            
            layout=layout)
        
    plot(fig, filename = filename + ".html",  auto_open=True)

def detect_anomalies_with_prediction_actuals(df, window = 20):
    '''
        Steps for detecting anomalies:
        1. Compute the error term(actual- predicted).
        2. Compute the rolling mean and rolling standard deviation(window is a week).
        3. Classify data with an error of 1.5,1.75 and 2 standard deviations as limits for low,medium and high anomalies. 
    '''
    df.replace([np.inf, -np.inf], np.NaN, inplace=True)
    df.fillna(0,inplace=True)
    df['error']=df['actuals']-df['predicted']
    df['percentage_change'] = ((df['actuals'] - df['predicted']) / df['actuals']) * 100
    df['meanval'] = df['error'].rolling(window=window).mean()
    df['deviation'] = df['error'].rolling(window=window).std()

    print "@Hongmin Wu: fill the NaN values as the first point"
    df['meanval'].fillna(df['meanval'][window-1], inplace=True) 
    df['deviation'].fillna(df['deviation'][window-1], inplace=True)   
    
    print "finished caculating the error, mean and std"
    
    df['-3s'] = df['meanval'] - (2 * df['deviation'])
    df['3s'] = df['meanval'] + (2 * df['deviation'])
    df['-2s'] = df['meanval'] - (1.75 * df['deviation'])
    df['2s'] = df['meanval'] + (1.75 * df['deviation'])
    df['-1s'] = df['meanval'] - (1.5 * df['deviation'])
    df['1s'] = df['meanval'] + (1.5 * df['deviation'])
    cut_list = df[['error', '-3s', '-2s', '-1s', 'meanval', '1s', '2s', '3s']]
    cut_values = cut_list.values
    cut_sort = np.sort(cut_values)
    df['impact'] = [(lambda x: np.where(cut_sort == df['error'][x])[1][0])(x) for x in
                               range(len(df['error']))]

    print "fininshed definied the impacts of errors"
    
    severity = {0: 3,
                1: 2,
                2: 1,
                3: 0,
                4: 0,
                5: 1,
                6: 2,
                7: 3}
        
    region = {0: "NEGATIVE",
              1: "NEGATIVE",
              2: "NEGATIVE",
              3: "NEGATIVE",
              4: "POSITIVE",
              5: "POSITIVE",
              6: "POSITIVE",
              7: "POSITIVE"}

    df['color'] =  df['impact'].map(severity)
    df['region'] = df['impact'].map(region)
    df['anomaly_points'] = np.where(df['color'] == 3, df['error'], np.nan) #Return elements chosen from x or y depending on condition.
    df = df.sort_values(by='load_date', ascending=False)
    df.load_date = pd.to_datetime(df['load_date'].astype(str), format="%Y-%m-%d")    
    
    return df
