import os,sys
import datetime
import math

import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
# pd.reset_option('^display.', silent=True)
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
#import pydotplus
import pylab as pl
import seaborn as sns
sns.set_style("darkgrid")
sns.set(rc = {'figure.figsize':(20,10)})
import plotly.express as px
import plotly

####-----------------------------------------------------------------------------------------------------------------------
####-----------------------------------------------------------------------------------------------------------------------
####-----------------------------------------------------------------------------------------------------------------------
def sigma_limit(df, coef=3):
  data_mean, data_std = np.mean(df), np.std(df)
  LCL, UCL= data_mean - coef*data_std, data_mean + coef*data_std
  return (LCL, data_mean, UCL)

def sigma_limit_grpby(df, col, grpby_col=[],coef=3):
  if grpby_col!=[]:
    cls=df.groupby(grpby_col)[col].apply(sigma_limit, coef=coef)
  else:
    cls=sigma_limit(df[col], coef=coef)
  # cls=pd.DataFrame(cls.tolist(), index=cls.index, columns=[f'L3S_{col}', f'AVG_{col}', f'U3S_{col}']).rename_axis(cls.index.name)
  cls=pd.DataFrame(cls.tolist(), index=cls.index, columns=[f'LL_{col}', f'AVG_{col}', f'UL_{col}']).rename_axis(cls.index.name)
  return cls

def sigma_limit_cols_grpby(df, cols, grpby_col, coef=3):
  cls_all=pd.DataFrame()
  for col in cols:
    cls  =sigma_limit_grpby(df , col=col, grpby_col=grpby_col, coef=coef)
    cls_all=pd.concat([cls_all, cls], axis=1)
  cls_all=cls_all.reset_index()
  return cls_all

def i_mr_sigma_limits(df, cols, grpby_col):
  I_chart_CLs=sigma_limit_cols_grpby(df,
                                    cols=cols,
                                    grpby_col=grpby_col).set_index('machine')

  MR_recovery=df.copy()
  MR_recovery[cols]=MR_recovery.groupby(grpby_col)[cols].diff(axis=0).abs()
  MR_chart_CLs=sigma_limit_cols_grpby(MR_recovery,
                                      cols=cols,
                                      grpby_col=grpby_col).set_index('machine')

  I_chart_CLs.columns=[f'I_{col}' for col in I_chart_CLs.columns]
  MR_chart_CLs.columns=[f'MR_{col}' for col in MR_chart_CLs.columns]

  sigma_lmt=pd.concat([I_chart_CLs,MR_chart_CLs],axis=1).reset_index()
  return sigma_lmt


####-----------------------------------------------------------------------------------------------------------------------
####-----------------------------------------------------------------------------------------------------------------------
####-----------------------------------------------------------------------------------------------------------------------
##TODO: add to common_funcs:
def control_limit(df):
  x_avg=np.mean(df)
  MR_avg=np.mean(df.diff().abs())
  
  x_LCL=x_avg-MR_avg*2.66
  x_UCL=x_avg+MR_avg*2.66
  
  MR_LCL=0
  MR_UCL=MR_avg*3.267 

  return (x_LCL, x_avg, x_UCL, MR_LCL, MR_avg, MR_UCL)

def control_limit_grpby(df, col, grpby_col=[]):
  if grpby_col!=[]:
    cls=df.groupby(grpby_col)[col].apply(control_limit)
  else:
    cls=control_limit(df[col])
  cls=pd.DataFrame(cls.tolist(), index=cls.index, columns=[
                                                          # f'I_LCL_{col}', f'I_AVG_{col}', f'I_UCL_{col}',
                                                          # f'MR_LCL_{col}', f'MR_AVG_{col}', f'MR_UCL_{col}',
                                                          f'I_LL_{col}', f'I_AVG_{col}', f'I_UL_{col}',
                                                          f'MR_LL_{col}', f'MR_AVG_{col}', f'MR_UL_{col}',                                                           
                                                           ]).rename_axis(cls.index.name)
  return cls

def i_mr_ctrl_limits(df, cols, grpby_col, coef):
  cls_all=pd.DataFrame()
  for col in cols:
    cls  =control_limit_grpby(df , col=col, grpby_col=grpby_col, coef=coef)
    cls_all=pd.concat([cls_all, cls], axis=1)
  cls_all=cls_all.reset_index()
  return cls_all

####-----------------------------------------------------------------------------------------------------------------------
####-----------------------------------------------------------------------------------------------------------------------
####-----------------------------------------------------------------------------------------------------------------------
####-----------------------------------------------------------------------------------------------------------------------
####-----------------------------------------------------------------------------------------------------------------------
####-----------------------------------------------------------------------------------------------------------------------
def hist_plot(df, col, quantile_range=[.10,.90]):
  fig, ax = plt.subplots(figsize=(20,10))
  sns.histplot(ax=ax,
              data=df.reset_index(),
              x=col,
              kde=True)

  quan=[df[col].quantile(q) for q in quantile_range]
  for x_loc, q in zip(quan,quantile_range):
      ax.axvline(x=x_loc, color='blue', linestyle='--')
      ax.text(x_loc+.02, 1, f'quantile {q*100}%: {round(x_loc,1)}')

  # data_mean, data_std = np.mean(df), np.std(df)
  # LCL, UCL = data_mean - 3*data_std, data_mean + 3*data_std
  # cutter_values=[LCL,UCL]
  # print(cutter_values)
  # for xl in cutter_values:
  #   ax.axvline(x=xl, color='red', linestyle='-')
  #   ax.text(xl+.02, 100, f'3 STD: {xl}')

  return quan

def plot_I_MR(df_I, limits, x_col='TimeStamp', fig=None):                 
  import re
  import plotly.graph_objects as go
  import math
  from plotly.subplots import make_subplots
  
  df_MR=df_I.drop([x_col],axis=1).diff(axis=0).abs()
  df_MR[x_col]=df_I[x_col]

  if type(limits)==pd.core.series.Series:
    limits=limits.to_frame()

  if fig is None:
    fig = go.Figure()
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Individual Chart", "Moving Range Chart"),
                        shared_xaxes=True,
                        )
  cols=[col for col in df_I.columns if (col!=x_col)&(col!='TimeStamp')]

  for con, df_plot in enumerate([df_I, df_MR]): 
    prefix='I_' if con==0 else 'MR_'
    limits_sub=limits[limits.index.str.contains(prefix)]
    for y in cols: 
      trace = go.Scatter(
                        x=df_plot[x_col],
                        y=df_plot[y],
                        mode='markers',
                        name =  f'{prefix}{y}'
                      )
      fig.add_trace(trace,
                    row=con+1, col=1)
      # print(min(trace.x))

    for limit_col in limits_sub:
      limits_sub2=limits_sub[limit_col]
      for con2, line in enumerate(limits_sub2.index):
        linecolor='blue'
        name='avg'
        if (limit_col=='ctrl_lmt')& ('AVG' not in line):
          linecolor='red'
          name ='control_limit'
        if (limit_col=='sigma_lmt')& ('AVG' not in line): #linecolor, name='brown' ,'3Sigma_limit'
          linecolor='brown'
          name ='3Sigma_limit'

        fig.add_trace(go.Scatter(
                                  name=name,
                                  x = [min(trace.x), max(trace.x)],
                                  y = [limits_sub2.loc[line], limits_sub2.loc[line]],
                                  mode = "lines",
                                  line={'color':linecolor,
                                        'width':1, 
                                        'dash': 'dash'},
                                  showlegend=True if ('AVG' not in line) & (con2==0)& (con==0) else False
                              ),
                      row=con+1, col=1)
        
        # fig.add_hline(y=limits_sub2.loc[line], line_width=1, line_dash="dash", line_color=linecolor)
        
  fig.update_layout(
                  # title_text = f"Plant: {plant}, machine: {machine}",
                  # legend_title_text = "Data",
                  # showlegend=False,
                  # autosize=True,
                  width=1200,
                  height=800,
                  legend=dict( orientation="h",
                              yanchor="bottom",
                              y=1.02,
                              xanchor="right",
                              x=1)  ,   
                  )
  # fig.update_xaxes(title_text=x_title)
  fig.update_yaxes(title_text=', '.join(df_I.columns.drop(['TimeStamp',x_col]).tolist()))
  # fig.show()

  return fig

def plot_recovery_asMine(df_plot,
                         x_col='TimeStamp',
                        ):
  import re
  import plotly.graph_objects as go
  import math
  from plotly.subplots import make_subplots

  cols0=[col for col in df_plot.columns if (col!=x_col)&(col!='TimeStamp')]
  # print(cols0)
  fig = make_subplots(rows=1, cols=1,
                      shared_xaxes=True,)

  cols0_line0={'color':'cornflowerblue',
                'width':1, 
                'dash': 'solid'} 

  cols0_line1={'color':'chocolate',
                'width':1, 
                'dash': 'solid'} 
  
  cols0_line2={'color':'darkcyan', 
              'width':1, 
              'dash': 'solid'} 
   
  mode='markers' if x_col=='TimeStamp' else 'markers+lines'
  # print(cols0)
  for con, col in enumerate(cols0):
    suffix=col.split('_')[-1]
    fig.add_trace(go.Scatter(x=df_plot[x_col], y = df_plot[col],            
                              mode = mode, line=eval(f'cols0_line{con}'), 
                              name = f"Recovery_{suffix}" if suffix!='asMine' else 'asMine',
                              showlegend=True,
                            ),
                  row=1, col=1)

  fig.update_layout(
                  # legend_title_text = "Data",
                  # showlegend=False,
                  # autosize=True,
                  width=1200,
                  height=800,
                  legend=dict( yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=.01
                            )
                  )
  # fig.update_xaxes(title_text=x_title)
  fig.update_yaxes(title_text=col)

  # fig.update_traces(mode="markers+lines", hovertemplate="TimeStamp: |%B %d, %Y")
  return fig
                               