import numpy as np
import pandas as pd
import logging
import time
import re

import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
import plotly.express as px

#import pydotplus
import pylab as pl
import seaborn as sns
import os,sys
import datetime as dt
import math

import plotly.express as px

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------Basic functions-------------------------------
def inWithReg(regLst, LstAll):
    """ search regualr expression list in a list

    Parameters:
    ----------
    regLst (list of strings with regx)

    LstAll (list of strings to be searched for regLst

    returns:
    -------
    out: the subset of LstAll which met contains any subset of regLst
    
    ind: a list of True/False values of existence LstAll in any subset of regLst

    Example:
    ---------
        regLst=['.vol_flag$','fefefre','_date']
        LstAll=['bi_alt_account_id', 'snapshot_date', 'snapshot_year','tv_vol_flag', 'phone_vol_flag']
        out,ind=inWithReg(regLst,LstAll)
        out=['tv_vol_flag', 'phone_vol_flag', 'snapshot_date']
        ind=[False,  True, False,  True,  True]

    -------
    Author: Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """
    out = []
    if type(regLst) != list:
        regLst = [regLst]
    for i in regLst:
        tmp = list(filter(re.compile(i).search, LstAll))
        out = out + tmp
    ind = np.in1d(LstAll, out)
    return out, ind

def retrieve_name(var):
    import inspect
    """Getting the name of a variable as a string
    Ref: https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string 
    """
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

def flattenList(ulist):
    """ makes a flat list out of list of lists

    Parameters:
    ----------
    ulist: a list of nested lists
    
    returns:
    -------
    results  ulist: a list of flatten ulist

    -------
    Author: Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """
    results = []
    for rec in ulist:
        if isinstance(rec, list):
            results.extend(rec)
            results = flattenList(results)
        else:
            results.append(rec)
    return results

def unique_list(seq):
    """ Get unique values from a list seq with saving the order of list elements
    Parameters:
    ----------
    seq: a list with duplicates elements
    -------
    out  (list): 
    a list with unqiue elements
    -------
    Author: Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """
    seen = set()
    seen_add = seen.add
    out=[x for x in seq if not (x in seen or seen_add(x))]
    return out
  
def check_timestamps(start, end, format_required='%Y-%m-%d'):
    '''validate the format required for the query'''
    try:
        check_start = type(time.strptime(start, format_required))
        check_end = type(time.strptime(end, format_required))
        if (check_start.__name__=='struct_time') & (check_end.__name__=='struct_time'):
            return True
    except ValueError as e:
        print(e)

def check_path(path):
    """Raise exception if the file path doesn't exist."""
    import argparse
    if '~' in path:
        path = os.path.expanduser(path)
    if not os.path.exists(path):
        msg = "File (%s) not found!" % path
        raise argparse.ArgumentTypeError(msg)
    return path

def pass_days(start_date, end_date):
    ##TODO: add comment
    import pandas as pd

    # if (start_date is None)|(end_date is None)| (pd.isnull(start_date))| (pd.isnull(end_date)):
    #   return None
    month_year_index = (
        pd.date_range(start=start_date, end=end_date, freq="D").to_period("Q").unique()
    )
    # print(start_date, end_date, month_year_index)

    pass_days_dict = {}
    for month_year in month_year_index:
        days_in_month = (
            min(end_date, month_year.end_time)
            - max(start_date, (month_year.start_time))
        ).days + 1
        pass_days_dict[month_year] = days_in_month

    result_series = pd.Series(pass_days_dict)
    qs = "Q" + result_series.index.quarter.astype(str)
    result_series = result_series.groupby(qs).sum()
    # print(result_series)
    return result_series.fillna(0)

def copy_ymls(dsToolbox, platform='databricks', destination=None):
  ##TODO: add comments:  
  import sys, os
  from io_funcs import io_funcs
  upath=dsToolbox.__file__
  if destination==None:
    destination=os.getcwd()
  for ufile in ['config.yml', 'sql_template.yml']:
    ufile_src=os.path.join(os.path.dirname(upath), ufile)
    ufile_desc=os.path.join(destination, ufile)
    ufile_desc_tmp=os.path.join(destination, f'.{ufile}.crc')
    print(f"copying {ufile_src} ---> {ufile_desc}")
    if platform=='databricks':
      dbutils=io_funcs.get_dbutils()
      dbutils.fs.cp(f'file://{ufile_src}', f'file://{ufile_desc}')
      dbutils.fs.rm(f'file://{ufile_desc_tmp}')

def remove_extra_none(nested_lst):
    items=list(dict.fromkeys(nested_lst)) 
    if (('None' in items) & (len(items)>1)):
        items.remove('None')  
    # print(items)
    return items

def setOutputFolder(outputFolder, uFiles, overWrite):
    """ creates output directory and copies template file(s)

    Parameters:
    ----------
    outputFolder (string): path of the outputFolder

    uFiles (a list of strings): of files that shoud be copied in outputFolder

    overWrite (True/False): if it True , it copies uFiles to outputFolder,even they exist;

    returns:
    -------
    outputFolder string
    """
    import os, shutil, sys
    
    # Setting output directory and copying template file(s)
    if len(outputFolder.split('/')) == 1:
        outputFolder = os.path.abspath(os.path.join(os.getcwd(), outputFolder))
    else:
        outputFolder = os.path.abspath(outputFolder)

    if os.path.exists(outputFolder) & (not overWrite):
        sys.exit(
            "Program terminated: overwrite is not allowed and the output directory exists")
    # elif os.path.exists(outputFolder):
    #     shutil.rmtree(outputFolder)
    #     os.makedirs(outputFolder)
    # else:
    #     os.makedirs(outputFolder)
    elif not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    # pdb.set_trace()
    if (overWrite):
        for uFile in uFiles:
            shutil.copyfile(
                uFile,
                os.path.join(
                    outputFolder,
                    os.path.basename(uFile)))
    return (outputFolder)
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# --------------------------------Date time -------------------------------
def readableTime(time):
    """ convert time to day, hour, minutes, seconds

    -------
    Author: Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """
    day = time // (24 * 3600)
    time = time % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time
    return (day, hour, minutes, seconds)

def datesList(range_date__year=[2018,2099],
              firstDate=None,
              lastDate=dt.datetime.now().date(),
              month_step=1):
  """generates a list of first dates of months within a given range of years
    Params:       
      range_date__year(list):  list of first and the last year+1                               
      firstDate (string or datetime): if it is not given, it will be the first day and month of range_date__year[0]
      lastDate (string or datetime): if it is not given,  it will be the current date
    Returns: python list 
  """ 
  import itertools  
  import datetime as dt

  # print(firstDate)
  # print(lastDate)
  yrs=[str(i) for i in range(range_date__year[0], range_date__year[1])]
  months=[str(i).zfill(2) for i in range(1,13, month_step)]
  udates=['-'.join(udate) for udate in itertools.product(yrs,months,['01'])]

  if isinstance(firstDate, str):
    print(firstDate)
    firstDate   = dt.datetime.strptime(firstDate, "%Y-%m-%d").date()
  elif isinstance(firstDate,pd._libs.tslibs.timestamps.Timestamp):
    firstDate   =firstDate.date()
  if isinstance(lastDate, str):
    lastDate     = dt.datetime.strptime(lastDate, "%Y-%m-%d").date()  
  elif isinstance(lastDate,pd._libs.tslibs.timestamps.Timestamp):
    lastDate   =lastDate.date()

  if firstDate is None:
    firstDate=dt.datetime.strptime(udates[0], '%Y-%m-%d').date() 
  if lastDate is None:
    lastDate=dt.datetime.strptime(udates[-1], '%Y-%m-%d').date() 

  if udates[-1]!=lastDate:
    udates.append(lastDate.strftime("%Y-%m-%d"))
  if udates[0]!=firstDate:
    udates[0]=firstDate.strftime("%Y-%m-%d")
    # udates.insert(0,firstDate.strftime("%Y-%m-%d"))
  
  udates=[ii for ii in udates if (dt.datetime.strptime(ii, '%Y-%m-%d').date()>=firstDate)&\
                                 (dt.datetime.strptime(ii, '%Y-%m-%d').date()<=lastDate)]
  # print(udates)
  return udates

def extract_start_end(udates, ii):
  import datetime as dt
  start_date=udates[ii]
  end_date=(dt.datetime.strptime(udates[ii+1], '%Y-%m-%d').date()- dt.timedelta(days=1)).strftime("%Y-%m-%d")
#   print(start_date,' to ',end_date ,":")  
  return start_date, end_date

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------Panadas functions------------------------------- 
def movecol(df, cols_to_move=[], ref_col='', place='After'):
    """ Reorders a panda dataframe columns
    Parameters:
    ----------
    df (pandas dataframe) 
    cols_to_move:                          list of columns to move
    ref_col(string):                       name of a specific column to move  cols_to_move columns to after/before it
    place(string) [options:"After","Before"]: cols_to_move columns will be move before/after it
    
    returns:
    -------
    df (pandas dataframe)  reordered dataframe

    -------
    Author:    https://towardsdatascience.com/reordering-pandas-dataframe-columns-thumbs-down-on-standard-solutions-1ff0bc2941d5
    """ 
    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    
    return(df[seg1 + seg2 + seg3]) 

def merge_between(df1, df2, groupCol, closed="both"):
    #   df1=df_pi_dic_wide
    #   df2=df_cases_edited
    #   groupCol='Vessel'

  df_out=pd.DataFrame(columns=df1.columns.tolist()+['Index_no'])
  for name, group_df in df1.groupby([groupCol]):
    print(name)
    df2_sub=df2.loc[df2[groupCol]==name]

    #     https://stackoverflow.com/questions/68792511/efficient-way-to-merge-large-pandas-dataframes-between-two-dates
    #     https://stackoverflow.com/questions/31328014/merging-dataframes-based-on-date-range
    #     https://stackoverflow.com/questions/69824730/check-if-value-in-pandas-dataframe-is-within-any-two-values-of-two-other-columns
    #     https://stackoverflow.com/questions/43593554/merging-two-dataframes-based-on-a-date-between-two-other-dates-without-a-common
    #     https://pandas.pydata.org/docs/reference/api/pandas.IntervalIndex.from_arrays.html
    i = pd.IntervalIndex.from_arrays(df2_sub['Start'],
                                     df2_sub['End'], 
                                     closed=closed
                                               )
    group_df['Index_no']=i.get_indexer(group_df['Date'])

    df_out=pd.concat([group_df, df_out],axis=0)
  
  return df_out

def cellWeight(df, axis=0):
  if axis==0:
    out=df.div(df.sum(axis=0), axis=1)
  else:
    out=df.div(df.sum(axis=1), axis=0)
  return out

def compare_dfs(df1,
                df2,
                df_names=['df1','df2']):

    uset=set(df1.columns).intersection(set(df2.columns))
    different_dtypes=[i  for i in uset if df1[i].dtype!=df2[i].dtype]

    if len(different_dtypes)!=0:
      print("Same columns have different data types:\n\t df1:\n", str(df1[different_dtypes].dtypes),"\n\t df2:\n",str(df2[different_dtypes].dtypes))
      convert_dict = {i:df1[i].dtype for i in different_dtypes} 
      df2 = df2.astype(convert_dict)
      print("df2 dtypes match with df1 dtypes")
      
    ##TOdo: debug it when there are columnd with same names in a df
    print(f"shape {df_names[0]}:",str(df1.shape))
    print(f"shape {df_names[1]}:",str(df2.shape))
    print(f"-----------------------------------------")    

    idx_only_df1=df1[~df1.index.isin(df2.index)]
    idx_only_df2=df2[~df2.index.isin(df1.index)]

    print(f"only rows in {df_names[0]}...")
    txt=idx_only_df1 if idx_only_df1.size!=0 else "Empty"
    print(txt)

    print(f"only rows in {df_names[1]}...")
    txt=idx_only_df2 if idx_only_df2.size!=0 else "Empty"
    print(txt)
    print(f"-----------------------------------------")    

    col_only_df1=df1.loc[:,~df1.columns.isin(df2.columns)]
    col_only_df2=df2.loc[:,~df2.columns.isin(df1.columns)]

    print(f"only columns in {df_names[0]}...")
    txt=col_only_df1 if col_only_df1.size!=0 else "Empty"
    print(txt)

    print(f"only columns in {df_names[1]}...")
    txt=col_only_df2 if col_only_df2.size!=0 else "Empty"
    print(txt)
    print(f"-----------------------------------------")        

    common_cols=df1.columns[df1.columns.isin(df2.columns)]
    common_idx=df1.index[df1.index.isin(df2.index)]

    df1_common=df1.loc[common_idx, common_cols]
    df2_common=df2.loc[common_idx, common_cols]

    df1_common_sub1=df1_common.select_dtypes(include=['number'])
    df2_common_sub1=df2_common.select_dtypes(include=['number'])
    common_bol_sub1=(abs(df1_common_sub1-df2_common_sub1).fillna(0))<.001

    df1_common_sub2=df1_common.select_dtypes(exclude=['number'])
    df2_common_sub2=df2_common.select_dtypes(exclude=['number'])
    common_bol_sub2=df1_common_sub2.fillna('')==df2_common_sub2.fillna('')

    common_bol=pd.concat([common_bol_sub1,common_bol_sub2],axis=1)

    #print("debug")
    #print(all(common_bol_sub1[debugcol]))
    #debugcol='co2_emission_t'
    #print(abs(df1_common_sub1[debugcol]-df2_common_sub1[debugcol]))    
    #print(pd.concat([df1_common_sub1.loc[~common_bol_sub1[debugcol],[debugcol,'machine_energy_usage']],
    #                 df2_common_sub1.loc[~common_bol_sub1[debugcol],[debugcol]]
     #              ],axis=1)
    #     )
    
    ###TOOD: all(common_bol) leads to the wrong result why???
    ##tmp=all((common_bol).all())
    df1N2per=round(common_bol.sum()/common_bol.shape[0]*100,0).sort_values()  
    if  (all(df1N2per==100))&(df1_common.size!=0):
        print("common rows and columns have same values, shape=",str(df1_common.shape))
        df1N2per=100
        df1N2diff=None
    elif (all(df1N2per==100))&(df1_common.size==0):
        print("no Common values- Hint: unify indexes")
        df1N2per=0
        df1N2diff=df1_common.compare(df2_common)
    else: 
        print("Percentage of common values:\n",df1N2per)
        idx=df1N2per!=100
        df1_common_diff=df1_common[df1N2per[idx].index.tolist()]
        df2_common_diff=df2_common[df1N2per[idx].index.tolist()]
        df1N2diff=df1_common_diff.compare(df2_common_diff)
        df1N2diff=df1N2diff.rename(columns={'self':df_names[0],'other':df_names[1]},level=1)
        print(df1N2diff)
      
    df1_out={"only rows in df1":idx_only_df1,
             "only columns in df1":col_only_df1,
            }

    df2_out={"only rows in df2":idx_only_df2,
         "only columns in df2":col_only_df2,
        }
    
    df1N2common={"Percentage of common values":df1N2per,
                 "comparing common columns with diffrenet values":df1N2diff,
                }
    
    return df1_out, df2_out, df1N2common

def cat2no(df):
    """ converts all categorical and object features of a dataframe (df) to cat.code

    Parameters:
    ----------
    df(pandas dataframe) with [sample*features] format

    thresh(number): thersold value for variance of features

    -------
    Author: Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """

    cat_columns = df.select_dtypes(include=['object']).columns.tolist()+df.select_dtypes(include=['category']).columns.tolist()
    if len(cat_columns) != 0:
        print('Categorical columns...\n'+cat_columns)
        df[cat_columns] = df[cat_columns].apply(
            lambda x: x.astype('category').cat.codes)
    return df

def reduce_mem_usage(df, obj2str_cols='all_columns', str2cat_cols='all_columns', verbose=False):
  """ iterate through all the columns of a dataframe and modify the data type
      to reduce memory usage.        
  """
  ## https://www.kaggle.com/code/konradb/ts-4-sales-and-demand-forecasting
  
  start_mem = df.memory_usage().sum() / 1024**2
  print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
  
  from pandas.api.types import is_datetime64_any_dtype as is_datetime

  for col in df.columns:
    col_type = df[col].dtype

    if ((str(col_type)[:3]=='float') |(str(col_type)[:3]=='int')): ##((col_type != object) & ~(is_datetime(df[col])) & (col_type!='str')):
      if verbose: print(col, ": compressing numeric column") 
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

    if  (col_type==object) & ((col in obj2str_cols)| (obj2str_cols=='all_columns')) : 
      df[col] = df[col].astype('str')
      obj2str=True
    else:
      obj2str=False

    if ((str(col_type)[:3]=='str')| obj2str) &  ((col in str2cat_cols)| (str2cat_cols=='all_columns')) :     ##~(is_datetime(df[col])):
      df[col] = df[col].astype('category')
      if (verbose) & (~obj2str):
        print(col, f": string --> category")
      if (verbose) & (obj2str):
        print(col, f": object --> string --> category")

  end_mem = df.memory_usage().sum() / 1024**2
  print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
  print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
  
  return df

def null_per_column(df):
    null_per = df.isnull().sum() / df.shape[0] * 100
    null_per = pd.DataFrame(null_per, columns=["null_percent"])
    null_per = (
        null_per.reset_index()
        .sort_values(by=["null_percent", "index"], ascending=False)
        .set_index("index")
    )
    return null_per

def unify_cols(df1, df2, df1_name, df2_name):
    df1.index=df2.index
    def unify_cols__sub(df1, df2, df1_name, df2_name):
        diff1=np.setdiff1d(df1.columns, df2.columns)
        if diff1.size!=0:
            print(f'Adding following columns to {df2_name} as there are in {df1_name}:\n {diff1}')
            df2=pd.concat([df2,
                  pd.DataFrame(0, index=df2.index, columns=diff1)], axis=1)
            df2=df2[df1.columns]
        return df2
    df2=unify_cols__sub(df1, df2, df1_name, df2_name)
    df1=unify_cols__sub(df2, df1, df2_name, df1_name)
    return df1, df2
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------EDA, Statisitcal analysis-----------------------     
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import f_classif

def compare_univar_fea(X, y, univar_fea_lst ):   
    ##TODO: refactor it: 
    arr = np.empty((0,X.shape[1]), float)
    for univar in univar_fea_lst:
        print(univar)
        uFunc=eval(univar)
        score = uFunc(X, y)
        if univar in ['mutual_info_classif','mutual_info_regression']:
          score=score
        elif univar in ['chi2']:
          score=score[1]
        elif univar in ['SelectFdr','SelectFdr','SelectFwe','f_classif']:
          ###TODO: correct it:
          selector = SelectKBest(uFunc,k='all')
          selector.fit(X,y)
          score = (selector.pvalues_)
    #     cols = selector.get_support(indices=True)

        if (score is  None):
            score=np.empty(X.shape[1]) 
            score[:]=np.nan
        arr = np.append(arr,[score] , axis=0)
    scores=pd.DataFrame(data=arr,index=univar_fea_lst,columns=X.columns)
    scores_long = pd.melt(scores.T.dropna(how='all').reset_index().rename(columns={"index": "feature"}),id_vars=['feature'],value_name='P_value', var_name='Feature_selection_Method')

    fig, ax = plt.subplots(figsize = (25,15))
    uplot   = sns.scatterplot(y="feature",
                          x="P_value",
                          hue="Feature_selection_Method",
                          style='Feature_selection_Method',
                          size='Feature_selection_Method',
                          data=scores_long,
                          ax=ax
                          ) 
    cutter_values=[.05]
    for con,xl in enumerate(cutter_values):
      ax.axvline(x=xl, color='red', linestyle='--')
      ax.text(xl, con+5, f'P_value={xl}',rotation=90, size=10)

    ## plt.xticks(rotation=90)
    plt.show()
    plt.close()

    return scores.T

def hypothesis_test(
                    df,
                    par,
                    group,
                    group_names,
                   ):
  import researchpy as rp
  
  df[group]=df[group].astype('bool')
  X1=df[par][df[group]]
  X2=df[par][~df[group]]
  
  group1_name, group2_name= group_names[0], group_names[1]
  des, res =rp.ttest(X1, X2,
                     group1_name= group1_name,
                     group2_name= group2_name,
                     equal_variances= False,
                     paired= False,
                     #correction= None
                    )
  res=res.set_index(res.columns[0])
  res.columns=[par]

  if res.loc['Two side test p value = '][0]!=0:
    txt=f"{par}: There is no difference between {group1_name} and {group2_name}"
    txt2='no difference'
  elif (res.loc['Two side test p value = '][0]==0) & (res.loc['Difference < 0 p value = '][0]==0):
    txt=f"{par}: {group1_name} is lower " #than {group2_name}"
    txt2='lower'
  elif  (res.loc['Two side test p value = '][0]==0) & (res.loc['Difference > 0 p value = '][0]==0):
    txt=f"{par}: {group1_name} is higher" #than {group2_name}"
    txt2='higher'
  else:
     txt2=txt=''
      
  res.loc['summary']=txt
  #   print(txt)

  summary=pd.DataFrame(txt2,index=[par],columns=[group1_name])
  #   print(summary)
  return des, res, summary

def hypothesis_test_batch_pars(df,
                              pars ,
                              group,
                              group_names):
  tsts=pd.DataFrame()
  stats=pd.DataFrame()
  summary=pd.DataFrame()
  for par in pars:
    par1_stats_tmp, par1_test_tmp, summary_tmp= hypothesis_test(df,
                                                                par=par,
                                                                group=group,
                                                                group_names=group_names
                                                               )
    stats=pd.concat([stats,
                    par1_stats_tmp],axis=0,
                    # keys=[par]
                    )
    
    tsts=pd.concat([tsts,
                    par1_test_tmp],axis=1)

    summary=pd.concat([summary,
                    summary_tmp],axis=0,
    #                     keys=[par]
                    )
  return stats, tsts, summary

def percent_agg(df, grpby1, grpby2, sumCol):
  agg1=df.groupby(grpby1)[sumCol].sum().reset_index()
  agg2=df.groupby(grpby2)[sumCol].sum().reset_index()

  agg1 = df.groupby(grpby1)[sumCol].sum()
  agg1 = agg1.groupby(level=grpby2).apply(lambda x:100 * x / float(x.sum())).reset_index()
  agg1.rename(columns={sumCol:f"{sumCol}_percent"}, inplace=True)
  
  agg1=agg1[agg1[f"{sumCol}_percent"]!=0]
  #   agg1=agg1.merge(agg2,on=grpby2)
  #   agg1[f'{sumCol}_percent']=np.round(agg1[f'{sumCol}_x']/agg1[f'{sumCol}_y']*100,0)
  #   agg1=agg1[agg1[outCol]!=0]

    ##NOTE:
  #   agg1.div(agg2, level=grpby2) * 100  doesnot work

  ##print(agg1.groupby(grpby2)[f"{sumCol}_percent"].sum())
  agg1[f"{sumCol}"]= pd.Series(df.groupby(grpby1)[sumCol].sum().values)
  
  return agg1

def rle_encode(data):
    #Ref:https://stackabuse.com/run-length-encoding/  
    encoding = ''
    prev_char = ''
    count = 1

    if not data: return ''

    for char in data:
        if char != prev_char:
            if prev_char:
                encoding += prev_char+"("+str(count) +");" 
            count = 1
            prev_char = char
        else:
            count += 1
    else:
        encoding += prev_char+"("+str(count) +");" 
        return encoding

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# ---------------------------Graph and plot functions----------------------
def corrmap(df0, method='kendall', diagonal_plot=True, **kwargs):
    """ plot a correlation heatmap matrix

    Parameters:
    ----------
    uData : (pandas dataframe) with [sample*features] format

    method : {‘pearson’, ‘kendall’, ‘spearman’} or callable
                pearson : standard correlation coefficient
                kendall : Kendall Tau correlation coefficient
                spearman : Spearman rank correlation
                callable: callable with input two 1d ndarray and returning a float.

    **kwargs: parameter of corr and seaborn heatmap: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html

    -------
    Author: Reza Nourzadeh- reza.nourzadeh@gmail.com
    """
    import inspect
    
    corr_args = list(inspect.signature(pd.DataFrame.corr).parameters)
    kwargs_corr = {k: kwargs.pop(k) for k in dict(kwargs) if k in corr_args}
    
    heatmap_args = list(inspect.signature(sns.heatmap).parameters)
    kwargs_heatmap = {k: kwargs.pop(k) for k in dict(kwargs) if k in heatmap_args}
    
    corr = df0.dropna(how='any',axis=0).drop_duplicates().corr(method=method,**kwargs_corr)
    # Generate a mask for the upper triangle
    
    if diagonal_plot:
      mask = np.zeros_like(corr)
      mask[np.triu_indices_from(mask)] = True
    else:
      mask=None

    plt.figure(figsize = (30,20))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    snsPlot = sns.heatmap(                                    
                            corr,
                            mask=mask,
                            cmap=cmap,
                            center=0,
                            square=True,
                            linewidths=.5,
                            cbar_kws={"shrink": .5},
                            fmt=".1f",
                            annot=True,
                            **kwargs_heatmap,
                            )
    figure = snsPlot.get_figure()
    # figure.savefig(os.path.join(outputFolder,"corr_map.png"), bbox_inches='tight')
    plt.show()
    plt.close()
    
    return corr, figure

def sankey(left, right, value, thershold, utitle, filename):
    """ create and plot a simplified sankey graph with only one source (left) and one target(right)

    Parameters:
    ----------
    left (array, shape) [n_samples]
    the label of left column

    right (array, shape) [n_samples]
    the label of right column

    value (array, shape) [n_samples]
    the value of transaction

    thershold (float)
    to filter those transactions which have less value than thershold

    utitle (string):
    the title of the plot

    filename (string):
    the location of the plot

    returns:
    -------
    tranactions(array, shape) [filtered n_samples*3]
    tranactions has three columns: left, right, value

    -------
    Author: - Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """

    tranactions0 = pd.concat(
        [left.rename('left'), right.rename('right'), value.rename('value')], axis=1)
    tranactions = tranactions0.groupby(
        ['left', 'right'], as_index=False).agg('sum')
    counts = tranactions0.groupby(
        ['left', 'right'], as_index=False).agg('count')
    tranactions = tranactions.loc[counts['value'] > thershold, :]
    tranactions.sort_values(['value'], ascending=[False], inplace=True)
    left = tranactions['left']
    right = tranactions['right']
    values = tranactions['value']

    #import chart_studio.plotly as py
    import plotly

    lbLeft = list(pd.unique(left))
    lbRight = list(pd.unique(right))

    # label=lbLeft+lbRight
    source = []
    target = []
    value = []
    for i in list(range(left.shape[0])):
        # if i==3:
        #     pdb.set_trace()
        tmpSource = np.where(
            np.asarray(lbLeft) == np.asarray(
                left.iloc[i]))[0].tolist()
        source = source + tmpSource

        tmpTarget = np.where(
            np.asarray(lbRight) == np.asarray(
                right.iloc[i]))[0].tolist()
        target = target + tmpTarget

        tmpValue = [values.iloc[i]]
        value = value + tmpValue

    target = [x + len(lbLeft) for x in target]

    data = dict(
        type='sankey',
        node=dict(
            pad=15,
            thickness=20,
            line=dict(
                color="black",
                width=0.5
            ),
            label=list(pd.unique(left)) + list(pd.unique(right))
            # color = ["blue", "blue", "blue", "blue", "blue", "blue"]
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))

    layout = dict(
        title=utitle,
        font=dict(
            size=10
        )
    )
    fig = dict(data=[data], layout=layout)
    plotly.offline.plot(fig, filename=filename)
    return tranactions

def explainedVar(pcaML, outputFile):
    """ calcluate and plot Variance Explained VS number of features for PCA
    ##TODO: add screeplot
    Parameters:
    ----------
    pcaML (float): Percentage of variance explained by each of the selected components.

    outputFile (string):
    the location of the plot

    returns:
    -------
    var  (float)
    cumulative varaince explained

    -------
    Author: - Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """
    
    eigen_values=pcaML.explained_variance_

    np.round(
            pcaML.explained_variance_ratio_,
            decimals=3)


    explained_var = np.cumsum(
        np.round(
            pcaML.explained_variance_ratio_,
            decimals=3) * 100)

    plt.ylabel('% explained_variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Analysis')

    plt.ylim(0, 100)
    plt.style.context('seaborn-whitegrid')
    plt.grid()
    plt.plot(explained_var)
    plt.savefig(outputFile, format='png', dpi=300, bbox_inches='tight')
    plt.close('all')

    return explained_var,eigen_values

def worldCloud_graph(txtSeries_df,outputFile):  
    ##TODO: document it  
    # txtSeries_df=tmp['prizm_68_2019']
    # outputFile=os.path.join(outputFolder,'enviroPostal_wordCloud2.png')

    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    
    if  type(txtSeries_df)==pd.core.series.Series:
        txt=' '.join(txtSeries_df.astype('str'))
        worldCloud_instance = WordCloud(width=800, height=400).generate(txt)
    else:
        worldCloud_instance = WordCloud(width=800, height=400).generate_from_frequencies(dict(zip(txtSeries_df.iloc[:,0] ,txtSeries_df.iloc[:,1])))

    ## Generate plot
    plt.figure(figsize=(20,10), facecolor='k')
    plt.imshow(worldCloud_instance)
    plt.axis("off")
    plt.savefig(outputFile, bbox_inches='tight')
    plt.close('all')
    # print("it was saved in "+os.path.join(outputFile))

def plot3D(udata, uY, xyzLabels, utitle, outPutFile):
    """ plot3d of udata

    parameters:
    ----------
    udata : (pandas dataframe) with [sample*3] format

    uY (string):
    the name of columns which is used to color dot plot

    xyzLabels: the list of string
    labels of Axis X,Y,Z

    utitle (string):
    the title of graph

    outputFile (string):
    the location of the plot

    -------
    Author: - Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """

    # %matplotlib notebook
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(xyzLabels[0], fontsize=15)
    ax.set_ylabel(xyzLabels[1], fontsize=15)
    ax.set_zlabel(xyzLabels[2], fontsize=15)
    ax.set_title(utitle, fontsize=20)
    targets = pd.unique(uY)
    colors = ['r', 'g', 'b', 'y']
    for target, color in list(zip(targets, colors)):
        indicesToKeep = uY.squeeze() == target
        ax.scatter(udata[indicesToKeep, 0], udata[indicesToKeep, 1],
                   udata[indicesToKeep, 2], c=color, s=50, alpha=.5)
    ax.legend(pd.unique(uY).astype('str'))
    ax.grid()

    plt.tight_layout()
    plt.savefig(outPutFile, format='png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print('Plot saved in ' + outPutFile)
    # for angle in list(range(0, 360,60)):
    #     ax.view_init(30, angle)
    #     # plt.draw()
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(outputFolder,fileName.split('.')[0]+str(angle)+'.'+fileName.split('.')[1]) , format='png', dpi=300, bbox_inches='tight')
    #     plt.pause(.001)

def lag_plot(x, y=None, nlags=24):
  # x=x.dropna()
  # print(f"data size after removing nulls({plant}):", x.shape )
  with sns.plotting_context("paper"):
    fig, ax = plt.subplots(nrows=math.ceil((nlags)/4), ncols=4, figsize=[15, 10])
    
    if y is None:
      fig.suptitle(f'Auto correlation plot {x.name}', fontsize=30)
      for i, ax_ in enumerate(ax.flatten()):
          
          pd.plotting.lag_plot(x, lag=i + 1, ax=ax_)
          # ax_.set_title(f"Lag {i+1}")
          ax_.ticklabel_format(style="sci", scilimits=(0, 0))
          ax_.set_ylabel(f"{x.name}$_t$")
          ax_.set_xlabel(f"{x.name}$_{{t-{i}}}$")
    else:
      fig.suptitle(f'Cross correlation plot {x.name} vs {y.name}', fontsize=30)
      for i, ax_ in enumerate(ax.flatten()):
          ax_.scatter(y=y, x=x.shift(periods=i), s=10)
          ax_.set_ylabel(f"{y.name}$_{{t}}$")
          ax_.set_xlabel(f"{x.name}$_{{t-{i}}}$")

    # plt.tight_layout()

def plot_ccf(x, y, lags,  ax=None, title="Cross-correlation", **kwargs):
  from statsmodels.tsa.stattools import ccf
  from matplotlib.ticker import MaxNLocator
  # Compute CCF and confidence interval
  cross_corrs = ccf(x, y, **kwargs)
  ci = 2 / np.sqrt(len(y))
  # Create plot
  if ax is None:
    fig, ax = plt.subplots(figsize=[10, 5])
  ax.stem(range(0, lags + 1), cross_corrs[: lags + 1])
  ax.fill_between(range(0, lags + 1), ci, y2=-ci, alpha=0.2)
  ax.set_title(title)
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  return ax, cross_corrs

def figures_to_html(figs, filename="dashboard.html"):
    """save a list of figures all to a single HTML file.
    """
    ##from https://stackoverflow.com/questions/45577255/plot-multiple-figures-as-subplots
    with open(filename, 'w') as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")
        
def save_plotly_fig(fig, fname_prefix, image_format="jpg"):
    import plotly      
    uFiles = [f"{fname_prefix}.{ext}" for ext in ["html", "json", image_format]]
    for uFile in uFiles:
        ext = uFile.split(".")[-1]
        if ext == "html":
            plotly.offline.plot(fig, filename=uFile, auto_open=False)
        elif ext == "json":
            plotly.io.write_json(fig, uFile)
        else:
            fig.write_image(uFile, width=2400, height=1400, scale=4)
    return uFiles

def cat2color_plotly(label_col, color_palette=None):
  import plotly.express as px
  # color_palette=px.colors.qualitative.Antique
  if color_palette is None:
    ## https://plotly.com/python/discrete-color/
    color_palette=px.colors.qualitative.Alphabet  ##Light24  ##Plotly
  domain=label_col.unique()
  if len(domain)>len(color_palette):
    print(f"number of available colors({len(color_palette)}) is more than categorizes({len(domain)}), change the palette")
  color_map = dict(zip(domain, color_palette[:len(domain)+2])) 
  c=label_col.map(color_map)
  return c, color_map

def plotly_group_stack(df_plot,
                        col2grp,
                        col2stack,
                        col2c,
                        date_col,
                        title,
                        color_palette=px.colors.qualitative.Light24,
                        patterns=['','/']
											):
  import plotly.graph_objects as go
  import plotly.express as px
  import pandas as pd

  x = [list(df_plot[date_col].dt.date.values),list(df_plot[col2grp].values)]

  colors,color_map=cat2color_plotly(df_plot[col2grp], color_palette=color_palette)

  fig = go.Figure()

  for shift_name, upattern in zip(df_plot[col2stack].unique(), patterns):
    df_tmp = df_plot.mask(df_plot[col2stack]!=shift_name, pd.NA)
    for Machine in df_plot[col2grp].unique():
      y = df_tmp[col2c].mask(df_plot[col2grp]!=Machine, pd.NA)
      fig.add_bar(
                  x=x,
                  y=y,
                  name=f"{Machine} - {shift_name}", 
                  hovertext = df_plot[col2stack],
                  marker_color=color_map[Machine],
                  marker_pattern_shape=upattern,
                  legendgroup=shift_name,
                  legendgrouptitle_text=shift_name,
                  hovertemplate="Date: %{x[0]}<br>"+
                                "Machine: %{x[1]}<br>"+
                                "Efficiency: %{y}<br>"+
                                "DayNight: %{hovertext}<br>",
                )	

  fig.update_layout(
                    barmode="relative",
                    xaxis_title="Date",
                    yaxis_title=col2c,
                    legend_title_text='',
                    title=title
                    )
  return fig

def subplot_plExpress(figs, sub_titles, main_title):
  from plotly.subplots import make_subplots
  figure_traces=[]
  for con, fig_sub in enumerate(figs):
    figure_traces_sub=[]
    for trace in range(len(fig_sub["data"])):
        if con>0: 
          fig_sub["data"][trace]['showlegend'] = False 
        figure_traces_sub.append(fig_sub["data"][trace])
    figure_traces.append(figure_traces_sub)
  figure = make_subplots(rows = 3, cols = 1, subplot_titles =sub_titles)
  figure.update_layout(height = 500, width = 1200, title_text =main_title, title_font_size = 25)
  for con, figure_traces_sub in enumerate(figure_traces):
    for traces in figure_traces_sub:
        figure.append_trace(traces, row = con+1, col = 1)
  return figure

from typing import Optional, Dict, Any, Tuple, Literal
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import scipy
from dataclasses import dataclass
from typing import List

@dataclass
class PlotConfig:
    """Configuration class for plot customization"""
    height: int = 800
    width: int = 1200
    colors: List[str] = None
    theme: str = 'white'  # or 'dark'
    title_font_size: int = 24
    axis_font_size: int = 14
    bins: int = 20
    boxpoints: Literal['all', 'outliers', False] = 'outliers'
    kde_points: int = 100
    jitter: float = 0.3
    violin_points: bool = True
    first_plot: Literal['box', 'violin'] = 'box'
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        if self.first_plot not in ['box', 'violin']:
            raise ValueError("first_plot must be either 'box' or 'violin'")

def validate_input(data: pd.DataFrame, dependent_var: str, group_var: str) -> None:
    """Validate input data and variables"""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame")
        
    if dependent_var not in data.columns:
        raise ValueError(f"Column '{dependent_var}' not found in DataFrame")
        
    if group_var not in data.columns:
        raise ValueError(f"Column '{group_var}' not found in DataFrame")
        
    if not pd.api.types.is_numeric_dtype(data[dependent_var]):
        raise ValueError(f"Column '{dependent_var}' must be numeric")
        
    if len(data[group_var].unique()) > 10:
        raise ValueError("Too many groups (>10) for meaningful visualization")

def plot_distributions(
    data: pd.DataFrame,
    dependent_var: str,
    group_var: str,
    config: Optional[PlotConfig] = None
) -> go.Figure:
    """
    Create interactive distribution plots for each group using plotly
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame containing the data
    dependent_var : str
        Name of the column containing the dependent variable
    group_var : str
        Name of the column containing the grouping variable
    config : PlotConfig, optional
        Configuration object for customizing the plots
        
    Returns:
    --------
    plotly figure object
    """
    # Input validation
    validate_input(data, dependent_var, group_var)
    
    # Use default config if none provided
    if config is None:
        config = PlotConfig()

    # Create subplot figure with 4 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'{config.first_plot.capitalize()} Plot', 
            'Histogram with KDE',
            'Empirical CDF',
            'Q-Q Plot'
        )
    )
    
    groups = sorted(data[group_var].unique())
    
    # 1. Box or Violin plot (top-left)
    for i, group in enumerate(groups):
        group_data = data[data[group_var] == group][dependent_var]
        if config.first_plot == 'box':
            fig.add_trace(
                go.Box(
                    y=group_data,
                    name=group,
                    boxpoints=config.boxpoints,
                    jitter=config.jitter,
                    pointpos=-1.8,
                    marker_color=config.colors[i % len(config.colors)],
                    marker=dict(size=6),
                    legendgroup=group,
                    showlegend=True
                ),
                row=1, col=1
            )
        else:  # violin plot
            fig.add_trace(
                go.Violin(
                    y=group_data,
                    name=group,
                    box_visible=True,
                    meanline_visible=True,
                    points="all" if config.violin_points else None,
                    marker_color=config.colors[i % len(config.colors)],
                    legendgroup=group,
                    showlegend=True
                ),
                row=1, col=1
            )

    # 2. Histogram with KDE (top-right)
    for i, group in enumerate(groups):
        group_data = data[data[group_var] == group][dependent_var]
        
        # Calculate KDE with proper scaling
        kde_x = np.linspace(group_data.min(), group_data.max(), config.kde_points)
        kde = scipy.stats.gaussian_kde(group_data)
        kde_y = kde(kde_x)
        
        # Scale KDE to match histogram height
        hist, bin_edges = np.histogram(group_data, bins=config.bins)
        scaling_factor = max(hist) / max(kde_y)
        kde_y = kde_y * scaling_factor
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=group_data,
                name=group,
                opacity=0.7,
                nbinsx=config.bins,
                marker_color=config.colors[i % len(config.colors)],
                legendgroup=group,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add KDE line
        fig.add_trace(
            go.Scatter(
                x=kde_x,
                y=kde_y,
                name=group,
                line=dict(color=config.colors[i % len(config.colors)]),
                legendgroup=group,
                showlegend=False
            ),
            row=1, col=2
        )

    # 3. ECDF (bottom-left)
    for i, group in enumerate(groups):
        group_data = data[data[group_var] == group][dependent_var]
        
        # Calculate ECDF
        sorted_data = np.sort(group_data)
        n = len(sorted_data)
        ecdf = np.arange(1, n + 1) / n
        
        fig.add_trace(
            go.Scatter(
                x=sorted_data,
                y=ecdf,
                name=group,
                mode='lines',
                line=dict(color=config.colors[i % len(config.colors)]),
                legendgroup=group,
                showlegend=False
            ),
            row=2, col=1
        )

    # 4. Q-Q Plot (bottom-right)
    for i, group in enumerate(groups):
        group_data = data[data[group_var] == group][dependent_var]
        qq = scipy.stats.probplot(group_data, dist="norm")
        
        # Add Q-Q points
        fig.add_trace(
            go.Scatter(
                x=qq[0][0],
                y=qq[0][1],
                mode='markers',
                name=group,
                marker=dict(color=config.colors[i % len(config.colors)]),
                legendgroup=group,
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Add Q-Q line
        z = np.polyfit(qq[0][0], qq[0][1], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=qq[0][0],
                y=p(qq[0][0]),
                mode='lines',
                name=group,
                line=dict(color=config.colors[i % len(config.colors)], dash='dash'),
                legendgroup=group,
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=config.height,
        width=config.width,
        title=dict(
            text="Distribution Analysis",
            x=0.5,
            font=dict(size=config.title_font_size)
        ),
        template=f"plotly_{config.theme}",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        boxmode='group',
        violinmode='group'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text=group_var, row=1, col=1, title_font=dict(size=config.axis_font_size))
    fig.update_xaxes(title_text=dependent_var, row=1, col=2, title_font=dict(size=config.axis_font_size))
    fig.update_xaxes(title_text=dependent_var, row=2, col=1, title_font=dict(size=config.axis_font_size))
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2, title_font=dict(size=config.axis_font_size))
    
    fig.update_yaxes(title_text=dependent_var, row=1, col=1, title_font=dict(size=config.axis_font_size))
    fig.update_yaxes(title_text="Count", row=1, col=2, title_font=dict(size=config.axis_font_size))
    fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1, title_font=dict(size=config.axis_font_size))
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2, title_font=dict(size=config.axis_font_size))
    
    return fig

def analyze_categorical_data(data, independent_var, dependent_var, alpha=0.05):
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2_contingency
    import matplotlib.pyplot as plt
    import seaborn as sns

    """
    Perform statistical analysis on categorical variables.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the variables
    independent_var : str
        Name of the independent variable column
    dependent_var : str
        Name of the dependent variable column
    alpha : float, optional
        Significance level for hypothesis testing (default is 0.05)
    
    Returns:
    --------
    dict
        Dictionary containing test results, contingency table, and visualization
    """
    # Create contingency table
    contingency_table = pd.crosstab(data[independent_var], data[dependent_var])
    
    # Perform Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate Cramer's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))
    
    # Create result dictionary
    results = {
        'contingency_table': contingency_table,
        'chi_square_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'cramers_v': cramer_v,
        'significant': p_value < alpha
    }
    
    # Add interpretation
    results['interpretation'] = interpret_results_analyze_categorical(results, alpha)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency_table, annot=True, cmap='YlOrRd', fmt='d')
    plt.title(f'Contingency Table: {independent_var} vs {dependent_var}')
    plt.tight_layout()
    
    return results

def interpret_results_analyze_categorical(results, alpha):
    """
    Interpret the statistical test results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing test results
    alpha : float
        Significance level
    
    Returns:
    --------
    str
        Interpretation of results
    """
    interpretation = []
    
    # Chi-square test interpretation
    if results['p_value'] < alpha:
        interpretation.append(f"There is a statistically significant relationship between the variables (p-value = {results['p_value']:.4f} < {alpha}).")
    else:
        interpretation.append(f"There is no statistically significant relationship between the variables (p-value = {results['p_value']:.4f} > {alpha}).")
    
    # Cramer's V interpretation
    cramers_v = results['cramers_v']
    if cramers_v < 0.1:
        strength = "negligible"
    elif cramers_v < 0.3:
        strength = "weak"
    elif cramers_v < 0.5:
        strength = "moderate"
    else:
        strength = "strong"
    
    interpretation.append(f"The strength of the association is {strength} (Cramer's V = {cramers_v:.3f}).")
    
    return " ".join(interpretation)
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# --------------------------Reading a SQL file/string----------------------
def split_sql_expressions_sub(text):
    """ split sql queries based on ";"

    Parameters:
    ----------
    text (string): (sql queries)

    returns:
    a list of queries
    --------

    -------
    Author: - Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """
    # from riskmodelPipeline.py
    results = []
    current = ''
    state = None
    for c in text:
        if state is None:  # default state, outside of special entity
            current += c
            if c in '"\'':
                # quoted string
                state = c
            elif c == '-':
                # probably "--" comment
                state = '-'
            elif c == '/':
                # probably '/*' comment
                state = '/'
            elif c == ';':
                # remove it from the statement
                current = current[:-1].strip()
                # and save current stmt unless empty
                if current:
                    results.append(current)
                current = ''
        elif state == '-':
            if c != '-':
                # not a comment
                state = None
                current += c
                continue
            # remove first minus
            current = current[:-1]
            # comment until end of line
            state = '--'
        elif state == '--':
            if c == '\n':
                # end of comment
                # and we do include this newline
                current += c
                state = None
            # else just ignore
        elif state == '/':
            if c != '*':
                state = None
                current += c
                continue
            # remove starting slash
            current = current[:-1]
            # multiline comment
            state = '/*'
        elif state == '/*':
            if c == '*':
                # probably end of comment
                state = '/**'
        elif state == '/**':
            if c == '/':
                state = None
            else:
                # not an end
                state = '/*'
        elif state[0] in '"\'':
            current += c
            if state.endswith('\\'):
                # prev was backslash, don't check for ender
                # just revert to regular state
                state = state[0]
                continue
            elif c == '\\':
                # don't check next char
                state += '\\'
                continue
            elif c == state[0]:
                # end of quoted string
                state = None
        else:
            raise Exception('Illegal state %s' % state)

    if current:
        current = current.rstrip(';').strip()
        if current:
            results.append(current)
    return results

def parse_sql_file(file_name):
    """ read sql queries in a file

    Parameters:
    ----------
    file (string): the location of sql file

    returns:
    a list of queries
    --------

    -------
    Author: - Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """
    # from riskmodelPipeline.py
    check_path(file_name)
    sql_statements = []
    with open(file_name, 'r') as f:
        for sql_statement in split_sql_expressions_sub(f.read()):
            sql_statements.append(sql_statement)
    return(sql_statements)

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------EDA, Statisitcal analysis-----------------------
def date2Num(df, dateCols):
    """ returns a new dataframe after deducing dates columns from the min dates of columns

    Parameters:
    ----------
    df (pandas dataframe) with [sample*features] format

    datecols  (a list of strings):: name of columns with date values

    returns:
    -------
    df[dateCols]-eval(df[dateCols].min()[0])

    -------
    Author: - Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """

    tmploc, _ = inWithReg(dateCols, df.columns.values)
        
    if len(tmploc) != 0:
        
        df[tmploc] = df[tmploc].astype('datetime64[ns]')

        print("conversion date features to number (month): date - " + str(df[tmploc].min()[0]))
        tmp3 = df[tmploc].apply(lambda x: x - df[tmploc].min()[0])
        tmp3 = tmp3.apply(lambda x: x / np.timedelta64(1, 'M'))
        df[tmploc] = tmp3.fillna(-1).round(0).astype('int64')
        df[tmploc]=df[tmploc].replace(-1,np.nan)

    return(df)

def find_low_variance(df, thresh=0.0):
    """ returns name of features  with variance less than thersold

    Parameters:
    ----------
    df(pandas dataframe) with [sample*features] format

    thresh(number): thersold value for variance of features

    -------
    Author: - Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """

    variance = df.var(skipna=True)
    low_variance = list(variance[variance <= thresh].index)
    return low_variance

def kruskalwallis2(x, y):
    """ calculate the Kruskal-Wallis H-test for independent samples

    Parameters:
    ----------
    x  array of observations
    y  array of groups

    returns:
    -------
    H-statistic  (float)
    The Kruskal-Wallis H statistic, corrected for ties
    p-value  (float)
    The p-value for the test using the assumption that H has a chi square distribution
    """
    groupednumbers = {}
    from scipy import stats
    for grp in y.unique():
        groupednumbers[grp] = x.values[y == grp]
    args = groupednumbers.values()
    tmp = stats.mstats.kruskalwallis(*args)
    # pdb.set_trace()
    return (tmp)

def chi2_contingency(x, y):
    """ Chi-square test of independence of variables in a contingency table.
    This function computes the chi-square statistic and p-value for the hypothesis test of independence of the observed frequencies in the contingency table.

    Parameters:
    ----------
    x  array of sample1

    y  array of sample2

    returns:
    -------
    p (float) :The p-value of the test

    Example:
    ---------

    -------
    Author: - Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """
    from scipy import stats
    xtab = pd.crosstab(x, y)
    pval = None
    if xtab.size != 0:
        try:
            _, pval, _, _ = stats.chi2_contingency(xtab)
        except Exception:
            pval = 0
    return pval

def corr_pointbiserial(binary_data, continuous_data, data):
    # TODO: correct it with nan
    """ computes the point biserial correlation of two pandas data frame columns

    Parameters:
    ----------
    binary_data :list : name of dichotomous data column

    continuous_data: list : name of dichotomous data column

    data (pandas dataframe) with [sample*feature] format: dataframe where above columns come from

    returns:
    -------
    out :  Point Biserial Correlation
    -------
    Author: - Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """

    import math
    bd_unique = data[binary_data].unique()
    
    g0 = data[data[binary_data] == bd_unique[0]][continuous_data]
    g1 = data[data[binary_data] == bd_unique[1]][continuous_data]
    
    s_y = np.std(data[continuous_data])
    n = len(data[binary_data])
    n0 = len(g0)
    n1 = len(g1)
    m0 = g0.mean()
    m1 = g1.mean()
    out=(m0-m1)*math.sqrt((n0*n1)/n**2)/s_y
    return out

def highcorr_finder(corrMat,df_scores,thershold):
    #TODO: add comment 
    corr_matrix = corrMat.abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than x
    drop_cols = [x for x in upper.columns if any(upper[x] >thershold)]
    drops_rows = [x for x in upper.index   if any(upper.loc[x] >thershold)]

    to_drop = list()
    mat_ind=list()
    for x,y in zip (drop_cols,drops_rows):
        tmp= y if df_scores[x]>df_scores[x] else x
        to_drop.append(tmp)
        mat_ind.append(x)
        mat_ind.append(y)

    high_corrs = dict(zip(drop_cols, drops_rows))

    return to_drop,mat_ind,high_corrs

def ortho_rotation(lam, method='varimax',gamma=None,
                    eps=1e-6, itermax=100):
    """
    ##TODO: document it 
    ## A VARIMAX rotation is a change of coordinates used in principal component analysis1 (PCA) that maximizes the sum of the variances of the squared loadings
    ## https://github.com/rossfadely/consomme/blob/master/consomme/rotate_factor.py
    Return orthogal rotation matrix
    TODO: - other types beyond 
    """
    if gamma == None:
        if (method == 'varimax'):
            gamma = 1.0
        if (method == 'quartimax'):
            gamma = 0.0

    nrow, ncol = lam.shape
    R = np.eye(ncol)
    var = 0

    for i in range(itermax):
        lam_rot = np.dot(lam, R)
        tmp = np.diag(np.sum(lam_rot ** 2, axis=0)) / nrow * gamma
        u, s, v = np.linalg.svd(np.dot(lam.T, lam_rot ** 3 - np.dot(lam_rot, tmp)))
        R = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < var * (1 + eps):
            break
        var = var_new

    return R

def discretizer(x,y,labels=["Q1", "Q2", "Q3","Q4"],method='cut'): 
    # print('discretizer:'+method)

    if method=='cut':
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()        
        out,bins=pd.cut(min_max_scaler.fit_transform(x.values.reshape(-1,1)).reshape(-1),bins=[-.1,.33,.66,1.1],labels=labels,retbins=True)

    elif method=='tree1':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(criterion = 'entropy',max_depth = 1)
        clf.fit(x.to_frame(),y)

        # if len(np.unique(clf.tree_.threshold[clf.tree_.threshold!=-2]))==1:
        #     # print ("max_depth increased")
        #     clf = DecisionTreeClassifier(criterion = 'entropy',max_depth = 2)
        #     clf.fit(x.to_frame(),y)

        bins =  np.sort(np.append(np.unique(clf.tree_.threshold[clf.tree_.threshold!=-2]),[x.max()+1/1e6,x.min()-1/1e6])).tolist()
        out=pd.cut(x,bins=bins,labels=labels[:(len(bins)-1)])

    elif method=='tree2':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(criterion = 'entropy',max_depth = 2)
        clf.fit(x.to_frame(),y)

        bins =  np.sort(np.append(np.unique(clf.tree_.threshold[clf.tree_.threshold!=-2]),[x.max()+1/1e6,x.min()-1/1e6])).tolist()
        out=pd.cut(x,bins=bins,labels=labels[:(len(bins)-1)])
    
    else:
        ##TODO: it needs to be corrected:
        out,bins=pd.qcut(x.rank(method='first'),q=3,labels=labels,retbins=True)
        # out,bins=pd.qcut(x,q=3,labels=labels,retbins=True, duplicates='drop')
        # out,bins=pd.qcut(x+ jitter(x),q=3,labels=labels,retbins=True)

    bins=[np.float(x) for x in bins]
    print(x.name+':\n'+str(bins))
    return (out ,bins)

def jitter(a_series, noise_reduction=1000000):
    # https://stackoverflow.com/questions/20158597/how-to-qcut-with-non-unique-bin-edges
    #TODO: add docs
    return (np.random.random(len(a_series))*a_series.std()/noise_reduction)-(a_series.std()/(2*noise_reduction))

def extract_equation(results_pars):
  vars=results_pars.reset_index()
  vars[0]=np.round(vars[0],2).astype(str)
  vars['ploys']=vars['index'].str.extract('np.power\((.+?),')
  vars['power']=np.where(vars['ploys'].isnull(),np.nan, vars['index'].str[-2:-1])
  vars['index']=np.where(vars['ploys'].isnull(),vars['index'],vars['ploys']+'**'+vars['power'])
  equation=""
  for row in vars.iterrows():
    sign='' if (np.sign(float(row[1][0]))==-1) or (row[0]==0) else '+'
    tmp=f"{sign}{row[1][0]}" if row[1]['index']=='Intercept' else f"{sign}{row[1][0]}*{row[1]['index']}"
    equation+=tmp
  return equation

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# --------------------------------loggers----------------------------------
def logmaker(uFile, name, logLevel=logging.INFO):
    """ create a logger

    Parameters:
    ----------
    uFile (string): path of logger file

    name (string): name of logger

    loggingLevel integer:
    The numeric values of logging levels are given in the following table. These are primarily of interest if you want to define your own levels, and need them to have specific values relative to the predefined levels. If you define a level with the same numeric value, it overwrites the predefined value; the predefined name is lost.
    
    -------
    #numerical logging level:
                # CRITICAL	50
                # ERROR	    40
                # WARNING	30
                # INFO	    20
                # DEBUG	    10
                # NOTSET	0

    returns:
    -------
    logger (logger type):  created logger

    -------
    Author: Reza Nourzadeh 
    """

    #   logging.basicConfig(filemode='w')

    # configure log formatter
    logFormatter1 = logging.Formatter(
        '%(asctime)s | %(levelname)-3s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logFormatter2 = logging.Formatter("%(message)s")
    #   logFormatter2 = logFormatter1

    # configure file handler
    fileHandler = logging.FileHandler(uFile, 'w')
    fileHandler.setFormatter(logFormatter1)

    # configure stream handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter2)

    # get the logger instance
    logger = logging.getLogger(name)

    # set the logging level
    logger.setLevel(logLevel)

    if not len(logger.handlers):
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)
    return logger

def custom_print(message, logger=None):
    print(message)
    if logger:
        logger.info(message)

class loggerWriter_sub:
    """ sub class used for loggerWriter2
    ###---------- Redirect stderr and stdout to log --------------
    ### https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python

    -------
    Author: Reza Nourzadeh 
    """

    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)

def loggerWriter2(stdoutLvl, stderrLvl):
    """ Redirect stderr and stdout to log

    Parameters:
    ----------
    stdoutLvl (string): the level of stdout

    stderrLvl (string): the level of stderrLvl

    returns:
    -------
    old_stdout,old_stderr: stdout and stderr before Redirecting to log

    -------
    Author: Reza Nourzadeh 
    """

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = loggerWriter_sub(stdoutLvl)
    sys.stderr = loggerWriter_sub(stderrLvl)
    return(old_stdout, old_stderr)