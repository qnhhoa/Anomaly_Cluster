import pandas as pd
from prompt_toolkit.key_binding.bindings.completion import E
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import datetime
import plotly.graph_objects as go
import plotly.express as px
import pickle

def log_to_hostname_csv(df, save_to):
  dff = df.copy()
  dff['Timestamp']=[':'.join(timestamp.split('.')[0].split(':')[:-1]) for timestamp in dff['Timestamp']]
  dff = dff[['hostid', 'MatrixID','MatrixName', 'delay', 'value', 'Timestamp']]

  hostnames = dff['hostid'].unique()
  list__df = [dff[dff['hostid']==hostnames[i]] for i in range(len(hostnames))]

  #save data to <hostname>.csv
  for i in range(len(hostnames)):
    list__df[i]= list__df[i].reset_index( drop=True)
    #list__df[i]= list__df[i].drop_duplicates(subset=['HostName','Timestamp'])
    list__df[i]['Timestamp'] = pd.to_datetime(list__df[i]['Timestamp'])
    list__df[i].to_csv(f'{save_to}/{hostnames[i]}.csv',index=None)

def check_features_before_after_resample(df,new_df,hostnames):
  compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
  for hostname in hostnames:
    print(compare(list(df[hostname].keys()), list(new_df[hostname].keys())))

def matrixname_row_to_column (df):
  features_dt = {}
  dff = df.copy()
  features = dff['MatrixName'].unique()
  dff = dff[['MatrixID','MatrixName', 'delay', 'value', 'Timestamp']]
  list__df = [dff[dff['MatrixName']==features[i]] for i in range(len(features))]
  for i in range(len(features)):
    list__df[i] = list__df[i].reset_index( drop=True)
    list__df[i] = list__df[i].rename(columns={'value': features[i]})
    list__df[i] = list__df[i].drop('MatrixName', axis=1)
    list__df[i] = list__df[i].drop_duplicates(subset=['Timestamp'])
    #list__df[i]['Timestamp'] = pd.to_datetime(list__df[i]['Timestamp'])
    #list__df[i]=list__df[i].set_index('Timestamp',drop=True)
    features_dt[features[i]] = list__df[i]
  return features_dt

def hostname_csv_to_feature_dict(hostnames, resample_time='60S'):
  hostname_feature_dict={}
  for hostname in hostnames:
    print(hostname)
    csv_directory = '/kaggle/working/datasets' #path to dataset
    hostname_df = pd.read_csv(f'{csv_directory}/{hostname}.csv', parse_dates=["Timestamp"])
    print(hostname_df)
    hostname_df["Timestamp"] = hostname_df["Timestamp"].dt.strftime('%m/%d/%Y %H:%M')
    # print(hostname_df.head(10))
    # hostname_df = hostname_df.drop(hostname_df.columns[0],axis=1)
    # print(hostname_df.head(10))
    hostname_df = hostname_df.dropna()
    hostname_features = matrixname_row_to_column(hostname_df)
    for feature in hostname_features.keys():
      dff = hostname_features[feature].copy()
      dff['Timestamp'] = pd.to_datetime(dff['Timestamp'])
      dff = dff.set_index('Timestamp',drop=True)
      #https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
      hostname_features[feature] = dff.resample(resample_time).interpolate('linear')
      #hostname_features[feature] = dff.resample('30S').mean().ffill()
    hostname_feature_dict[hostname] = hostname_features
  return hostname_feature_dict

def update_new_start_end_timestamp(df, hostnames, start_timestamp, end_timestamp, save_to_pickle, filename):
  new_df, new_df_, new_df_concat = {}, {}, {}
  for hostname in hostnames:
    new_feature = {}
    new_feature_ = {}
    for feature in df[hostname].keys():
      new_feature[feature]=df[hostname][feature].loc[start_timestamp:end_timestamp][[feature]]
      new_feature_[feature]=df[hostname][feature].loc[start_timestamp:end_timestamp]
      new_feature_[feature]['Timestamp'] = new_feature_[feature].index
      new_feature_[feature].index = np.arange(1, len(new_feature_[feature]) + 1)
    new_df[hostname]=new_feature
    new_df_[hostname]=new_feature_
    
    vis = pd.concat(new_df[hostname],axis=1)
    vis.columns = new_df[hostname].keys()
    new_df_concat[hostname]=vis

  if save_to_pickle:
    pickle_out = open(f'{filename}.pickle', 'wb')
    pickle.dump(new_df_, pickle_out)
    pickle_out.close()

    pickle_out = open(f'{filename}-concat.pickle', 'wb')
    pickle.dump(new_df_concat, pickle_out)
    pickle_out.close()
  else: pass
  return new_df_concat

def get_start_end_timestamp(df, hostnames):
  infor_df ={}
  for hostname in hostnames:
    index_list = {}
    infor = []
    for feature in df[hostname].keys():
      infor.append({
        'feature name': feature,
        # 'datapoint':  df[hostname][feature].shape[0],
        'start_time':  df[hostname][feature].index[0],
        'end_time':  df[hostname][feature].index[-1]
    })
      index_list[feature]=df[hostname][feature].index
    infor_df[hostname]= pd.DataFrame.from_records(infor)
#   len_feature_df = []
#   for hostname in hostnames:
#     print(hostname, infor_df[hostname]['datapoint'].unique())
#     len_feature_df.extend(infor_df[hostname]['datapoint'].unique())

#   print('**********************************************************************')
#   print(min(len_feature_df))
#   print('**********************************************************************')
  infor_dff = pd.concat(infor_df)
  start_ = [pd.to_datetime(time) for time in infor_dff['start_time'].unique()]
  end_ = [pd.to_datetime(time) for time in infor_dff['end_time'].unique()]
  start_timestamp = max(start_)
  end_timestamp = min(end_)
  return start_timestamp, end_timestamp


