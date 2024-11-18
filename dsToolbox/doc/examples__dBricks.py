# Databricks notebook source
# MAGIC %md
# MAGIC # Installing dsToolbox

# COMMAND ----------

###01-installing requirments:
%pip install -r /dbfs/FileStore/dsToolbox/requirements.txt

###02:
%pip install /dbfs/FileStore/dsToolbox/dsToolbox-0.2.0-py3-none-any.whl

##as a pre installed library in databricks cluster use:
# dbfs:/FileStore/dsToolbox/dsToolbox-0.2.0-py3-none-any.whl


# COMMAND ----------

# from importlib import reload  
import dsToolbox

import dsToolbox.io_funcs as io_funcs
import dsToolbox.common_funcs as common_funcs

# ??io_funcs.load_config

# COMMAND ----------

# MAGIC %md
# MAGIC # io_funcs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query Synapse 

# COMMAND ----------

query_str="""(SELECT dumpshiftid,
                    dumpshiftcrew,
                    LoadFullTimeStamp,
                    excavname,
                    calc_matgroup,
                    dispatchmaterialtype,
                    LoadSizeTon,
                    right(excavname,4) as equipment_id
        FROM mcs.vw_mcs_haul_cycle a with (nolock)
        WHERE CAST(DumpEmptyTimestamp AS date) between '2023-01-01'
                and '2023-01-15') query"""

sp_mcs_haul_cycle=io_funcs.query_synapse_db(
                                        query=query_str,
                                        key_vault_dict='azure_synapse',
                                        custom_config=None,
                                        verbose=True,
                                        )
                            
###using predefined queries:
replace_dict={'start___date':'2023-01-01',
              'end___date'  :'2023-01-15'
             }
sp_mcs_haul_cycle=io_funcs.query_template_run( 
                                        query_temp_name='vw_mcs_haul_cycle',
                                        replace_dict=replace_dict,
                                        # custom_config=config_dict,
                                        platform='databricks',
                                        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Customized sql template and configs

# COMMAND ----------

# MAGIC %md
# MAGIC #### Customized SQL templates 

# COMMAND ----------

# dsToolbox.config_dict
# dsToolbox.sql_template_dict

##copying config and sql_template yaml files to the current directory, to use them later as custom config or sql_template:
common_funcs.copy_ymls(dsToolbox, platform='databricks', destination=None)

replace_dict={'start___date':'2023-01-01',
              'end___date'  :'2023-01-15'
             }
sp_mcs_haul_cycle=io_funcs.query_template_run(
                                                query_temp_name='vw_mcs_haul_cycle',
                                                replace_dict=replace_dict,
                                                custom_sql_template_yml='/Workspace/Repos/reza.nourzadeh@gmail.com/ds_toolbox/dsToolbox/doc/sql_template.yml',
                                                # custom_config=config_dict,
                                                platform='databricks',
                                                )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Customized config

# COMMAND ----------

###Method 1:
##Step1-copying config and sql_template yaml files to the current directory, to use them later as custom config or sql_template:
common_funcs.copy_ymls(dsToolbox, platform='databricks', destination=None)

##Step2- modification config.yml

df_recovery_asB_date = io_funcs.blob2pd(blob_dict={'storage_account':'sadatasciencedmm',
                                                    'container'       :'recovery',
                                                    'blob'            :f'ml_exploring/recovery_15_new_nonAGG.parquet',
                                                    # 'blob'          :f'ml_exploring/recovery_15_new.parquet',
                                                    },
                                    custom_config='/Workspace/Repos/reza.nourzadeh@gmail.com/ds_toolbox/dsToolbox/doc/config.yml',
                                    parse_dates=['Date'],
                                    usecols=['Date','mbi','water'],
                                    verbose=True,
                                    platform='databricks'
                                    )

##---------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------
###Method 2:
CUSTOM_CONFIG_DICT = {
                      "key_vault_dictS": 
                          {'sadatasciencedmm': 
                              {"key_vault_name": "kv-ds-recovery-prd", 
                              "secret_name": 'sadatasciencedmm'
                              }
                          }
                    }
###or:
CUSTOM_CONFIG_DICT = {
                      'storage_account':'sadatasciencedmm',
                      "key_vault_name": "kv-ds-recovery-prd", 
                      "secret_name": 'sadatasciencedmm',
                      }


df_recovery_asB_date = io_funcs.blob2pd(blob_dict={'storage_account':'sadatasciencedmm',
                                                    'container'       :'recovery',
                                                    'blob'            :f'ml_exploring/recovery_15_new_nonAGG.parquet',
                                                    # 'blob'          :f'ml_exploring/recovery_15_new.parquet',
                                                    },
                                    custom_config=CUSTOM_CONFIG_DICT,
                                    parse_dates=['Date'],
                                    usecols=['Date','mbi','water'],
                                    verbose=True,
                                    platform='databricks'
                                    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading a file as a dataframe and writing df as a file in Delta Table:

# COMMAND ----------

# MAGIC %md
# MAGIC #### Spark Dataframe

# COMMAND ----------

start_date ='2023-07-01'
end_date   ='2023-07-17'

query_txt=f""" select timestamp, tag, value
                from pidata.pidelta_knifegate
                where timestamp between '{start_date}'
                                and '{end_date}'
                and
                tag like '%PV'
              --  and tag like 'kr%'                                
          """
df0=io_funcs.query_deltaTable_db(
                                # spark,
                                query=query_txt,
                                key_vault_dict='azure_synapse',
                                custom_config=None,
                                verbose=True
                                ).toPandas()

# COMMAND ----------

table_schema= f'tst_dsToolbox.sp_tst'
io_funcs.spark2deltaTable(
                          # spark,
                          sp_mcs_haul_cycle,
                          table_name    = table_schema.split('.')[1],
                          schema        = table_schema.split('.')[0],
                          write_mode = 'append',
                          mergeSchema=True,
                          partitionby   = ['YEAR', 'MONTH']
                        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Copying a file from delta Table to blob stroage

# COMMAND ----------

import os
ufile= '/dbfs/tmp/knifeGate_failed.html'
outputFolder= ''
io_funcs.dbfs2blob(
                  # spark, dbutils,
                  ufile,
                  blob_dict={'storage_account' :'knifegateprod',
                              'container'        :'knifegatever2',
                              'blob'             :os.path.join(outputFolder, os.path.basename(ufile))}
                  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading a file as a dataframe and writing a df as a file in blob storage 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Panads Dataframe

# COMMAND ----------

df_recovery_asB_date = io_funcs.blob2pd(blob_dict={'storage_account':'sadatasciencedmm',
                                                    'container'       :'recovery',
                                                    'blob'            :f'ml_exploring/recovery_15_new_nonAGG.parquet',
                                                    # 'blob'          :f'ml_exploring/recovery_15_new.parquet',
                                                    },
                                    parse_dates=['Date'],
                                    usecols=['Date','mbi','water'],
                                    verbose=True,
                                    # platform='local'
                                    )

# COMMAND ----------

udata_smp = io_funcs.blob2pd(blob_dict={'storage_account':'sadatasciencedmm',
                                          'container'       :'recovery',
                                          'blob'            :f'ml_exploring/df_recovery_udata.parquet'
                                          },
                              verbose=True)

# io_funcs.pd2blob(udata_smp,
#         blob_dict={'storage_account':'sadatasciencedmm',
#                   'container'       :'recovery',
#                   'blob'            :f'ml_exploring/df_recovery_udata.xlsx'
#                   },
#         platform='databricks',
#         sheetName='dataframe1',
#         # append=True, 
        
#       )

io_funcs.xls2blob({'dataframe1':udata_smp.sample(1000),
                  'dataframe2':udata_smp.sample(1000),
                  'dataframe3':udata_smp.sample(1000),
                  },
        blob_dict={'storage_account':'sadatasciencedmm',
                  'container'       :'recovery',
                  'blob'            :f'ml_exploring/df_recovery_udata.xlsx'
                  },
        platform='databricks',
       
      )

# io_funcs.pd2blob(udata_smp,
#         blob_dict={'storage_account':'sadatasciencedmm',
#                   'container'       :'recovery',
#                   'blob'            :f'ml_exploring/df_recovery_udata.csv'
#                   },
#         platform='databricks',
#         # sheetName='dataframe1',
#         index=False
#       )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Spark Dataframe

# COMMAND ----------

sp=io_funcs.blob2spark(
                      # spark,
                      blob_dict={'container':'knifegatever2', 
                                  'blob': f'df_agg_lag.parquet',
                                  'storage_account':'knifegateprod'},
                      )
pd_df=sp.toPandas()  

# COMMAND ----------

io_funcs.spark2blob(
                  # spark,
                   sp,
                   blob_dict={'container':'knifegatever2', 
                                      'blob': f'df_tst.parquet',
                                      'storage_account':'knifegateprod'},
                   write_mode="""mode('append')
                                 """
                  )

# COMMAND ----------

# MAGIC %md
# MAGIC #### non dataframe:

# COMMAND ----------

from importlib import reload  
reload(io_funcs)
for file in ['sample_demonstrations_20231009.pkl',
              'sample_28800.zip',
              'sample_best_model.h5' ,
              'sample_bestModelHigh.mat',
              'sample_hyperparameters.txt',
              'sample_scaler_X.joblib'
            ]:

  io_funcs.blob2pd(blob_dict={'storage_account':'sadatasciencedmm',
                              'container':'recovery',
                              'blob': f'New folder/{file}'
                            },
                  platform='databricks',
                  verbose=True)

f = open("/tmp/sample_hyperparameters.txt", "r")
print(f.read())
# dbutils.fs.ls("dbfs:/mnt/data/")
# dbutils.fs.rm("tmp/from_blob/")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading PI Web-API__under_edition

# COMMAND ----------

PI_WEB_API_TAGS = {
    'kr1_1120-121-FFIC-1310-1A.PV': 'k1_l1_owr',
    'kr1_1120-121-FFIC-1410-1A.PV': 'k1_l2_owr'
}

df=io_funcs.pi2pd_rawData(PI_WEB_API_TAGS.keys(),
                          start_date='2023-07-13', end_date='2023-08-10',
                          pi_vault_dict='webapi',
                          # platform='local'
                          )

# df=io_funcs.pi2pd_interpolate(PI_WEB_API_TAGS.keys(),
#                           start_date='2023-07-13', end_date='2023-08-10',
#                           pi_vault_dict='webapi',
#                           # platform='local'
#                           )

# COMMAND ----------

# MAGIC %md
# MAGIC # ML functions:

# COMMAND ----------

import pandas as pd
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
# pd.reset_option('^display.', silent=True)
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

udata = io_funcs.blob2pd(blob_dict={'storage_account':'sadatasciencedmm',
                                          'container'       :'recovery',
                                          'blob'            :f'ml_exploring/df_recovery_udata.parquet'
                                          },
                              verbose=True)
target='high_loss'
X=udata.drop([target], axis=1)
y=udata[target]    
RANDOM_STATE=42

un_splits=10
sk_fold  = StratifiedKFold(n_splits=un_splits, shuffle=True, random_state=RANDOM_STATE)
# sk_fold = TimeSeriesSplit(n_splits=un_splits)

def dataSpliter(udata, perc=.75):
  counts=udata['plant_K1'].value_counts()
  # print(counts)
  train_nos=int(udata.shape[0]*perc/2)
  udata_train=udata.groupby(['plant_K1']).head(train_nos)
  udata_test_k1=udata[udata['plant_K1']==1].tail(counts[1]-train_nos)
  udata_test_k2=udata[udata['plant_K1']==0].tail(counts[0]-train_nos)
  udata_test=pd.concat([udata_test_k1, udata_test_k2], axis=0)
  return udata_train, udata_test

udata_train, udata_test_val= dataSpliter(udata, perc=.75)
udata_val, udata_test= dataSpliter(udata_test_val, perc=.60)

X_train, y_train= udata_train.drop([target], axis=1), udata_train[target]
X_test_val,  y_test_val  = udata_test_val.drop([target], axis=1), udata_test_val[target]

X_test,  y_test = udata_test.drop([target], axis=1), udata_test[target]
X_val,   y_val  = udata_val.drop([target], axis=1), udata_val[target]

# COMMAND ----------

import dsToolbox.ml_funcs as ml_funcs
# reload(ml_funcs)

scores_names=['recall' ,
              'precision',
              'accuracy',
              'auc_weighted',
              # 'balanced_accuracy',
              # 'roc_auc',  
              # 'aucpr',
              
              'f1',
              'kappa',
              'mcc',
              ]

y_model_Xval, _ ,_ = ml_funcs.ml_prediction(XGBClassifier(eval_metric='aucpr',
                                                          # early_stopping_rounds=20,
                                                          n_estimators=200,
                                                          scale_pos_weight=float(np.sum(y == 0)) / np.sum(y==1),
                                                          random_state=RANDOM_STATE
                                                          ),
                                            X,
                                            y,
                                            sk_fold=sk_fold, #sk_fold, #[X_val,   y_val],
                                            # X_test=X_test,
                                            # y_test=y_test
                                            )

y_model=y_model_Xval
uscores, confMats= ml_funcs.classifer_performance_batch(y_model, 
                                            map_lbls={0:'Low Loss', 1:'High Loss'},
                                            scores_names=scores_names)
uscores

# COMMAND ----------

# MAGIC %md
# MAGIC # Draft:

# COMMAND ----------

# # %pip install pdoc3

# import os
# os.getcwd()
# dbutils.fs.rm('dbfs:/tmp/io_funcs.html',True)
# !pdoc3 --html --force -o /dbfs/tmp/ ../io_funcs.py
# io_funcs.dbfs2blob('/dbfs/tmp/io_funcs.html',
#                    blob_dict={'container':'knifegatever2', 
#                               'blob': f'io_funcs.html',
#                               'storage_account':'knifegateprod'}
#                    )


# COMMAND ----------

# import os
# os.chdir('C:\\Users\\rnourza\\main_folder\\codes\\dsToolbox')
# from importlib import reload  
# import dsToolbox.io_funcs as io_funcs
# reload(io_funcs)

# start_date ='2023-01-01'
# end_date   ='2023-07-17'

# query=f"""(SELECT *
#             FROM mcs.vw_mcs_haul_cycle a with (nolock)
#             WHERE DumpLocation LIKE '%cr%' 
#               AND CAST(DumpEmptyTimestamp AS date) between '{start_date}' and '{end_date}') query"""

# df=io_funcs.query_synapse_local(
#                                 query=query,
#                                 key_vault_dict ='azure_synapse',
#                                 verbose=True,
#                                 )


# COMMAND ----------

# weather_tags=['kr0_0020-TI-AC4WEATHER-1.PV',
#               'kr0_0020-LI-AC4WEATHER.PV',
#               'kr0_0020-SI-AC4WEATHER.PV',
#               'kr0_0020-PI-AC4WEATHER.PV',
#               'kr0_0020-AI-AC4WEATHER-1.PV',
#               'kr0_0020-TI-AC4WEATHER-3.PV',]

# start_date ='2023-01-01'
# end_date   ='2023-07-17'

# query_txt=f"""select tag,
#                     time as timestamp,
#                     CAST(value AS float) as value
#           from pidata.pidelta
#           where time between '{start_date}'
#                          and '{end_date}'
#                 and 
#                 upper(tag) like '%AC4WEATHER%' -- '%KR0_0020-%'
#           """

# df0=io_funcs.query_deltaTable(spark,
#                             query=query_txt,
#                             verbose=True
#                             ).toPandas()


# COMMAND ----------

# query_txt="""select tag,
#                     time as timestamp,
#                     CAST(value AS float) as value
#           from pidata.pidelta
#           where time between '2023-02-10'
#                          and '2023-05-17'
#                 and 
#                  regexp_like (upper(tag),
#                               'KR[12]_[12]110-112-WI-d{4}-[12].PV'                     
#                               )
#                 -- 'KR0_0020-[A-Z]{2}-AC4WEATHER.'                              
#                 and upper(tag) rlike ('KR[12]_[12]110-112-WI-d{4}-[12].PV')
#                 --upper(tag) like '%AC4WEATHER%' 
#                 --or 
#                 --upper(tag) like '%KR0_0020-%'
#           """

#           # 'kr1_1110-112-WI-1065-1.PV' :   'CR01',  
#           # 'kr2_2110-112-WI-1065-1.PV' :   'CR02',     
#           # 'kr1_1110-112-WI-1742-2_PV' :   'CR03', 
#           # 'kr2_2110-112-WI-1742-2.PV' :   'CR04'}

# query_txt=f"""select tag,
#                     time as timestamp,
#                     CAST(value AS float) as value
#           from pidata.pidelta
#           where time between '{start_date}'
#                          and '{end_date}'
#                 and 
#                 -- regexp_like (upper(tag),
#                 --              'KR0_0020-[A-Z]{2}-AC4WEATHER.'
#                 --              )
#                 -- and upper(tag) rlike 'KR0_0020-[A-Z]{2}-AC4WEATHER.'
#                 upper(tag) like '%AC4WEATHER%' -- '%KR0_0020-%'
#           """
