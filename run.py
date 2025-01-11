import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from deep_learning_models.woclsa import woclsa
from deep_learning_models.cnnlstma import cnnlstma
from deep_learning_models.ensemble import ensemble
import pandas as pd
from data_preprocessing.imputation import imputation
from data_preprocessing.normalisation_encoding import process_numerical_data, encode_non_numerical_data
from data_preprocessing.feature_selection.execute_fs import execute_fs
import keras
import tensorflow as tf
import datetime
import time
import os
import math
import numpy as np
import psutil

num_gpus = len(tf.config.list_logical_devices('GPU'))
print(f"Number of GPUs detected: {num_gpus}")


# Reading datasets
filename = "cleveland.csv"
fs_file = "./datasets/fs_heart.txt"
if filename == "balanced_dataset.xlsx":
  data = pd.read_excel('./datasets/' + filename)
else:
  data = pd.read_csv('./datasets/' + filename)
df = pd.DataFrame(data)


if filename == 'cleveland.csv':
  target = 'num'
  df[target] = df[target].apply(lambda x: 1 if x > 0 else x)
  df = df.apply(pd.to_numeric, errors='coerce')
elif filename == 'CVD_FinalData.csv':
  target = 'TenYearCHD'
  df = df.drop(columns=['id'])
elif filename == "heart.csv":
  target = 'HeartDisease'
elif filename == "balanced_dataset.xlsx":
  target = 'Label'
  df = df.drop(columns=["Patient ID"])
elif filename == "darwin_data.csv":
  target = 'class'
  df = df.drop(columns=["ID"])
  df[target] = df[target].map({'P': 1, 'H': 0})
elif filename == "toxicity_data.csv":
  target = 'Class'
  df[target] = df[target].map({'Toxic': 1, 'NonToxic': 0})
elif filename == "arcene_data.csv":
  target = 'labels'
  df[target] = df[target].replace({-1:0})
elif filename == "tuandromd.csv":
  target = 'Label'
elif filename == "period_changer.csv":
  target = 'Class'
elif filename == 'malicious_executable.csv':
  target = 'Label'
elif filename == 'semeion_dataset.csv':
  target = 'Label'

# MissForest Imputation
if df.isnull().values.any():
  df = imputation(df, target)

# Standardisation/Normalisation and One-Hot Encoding
df = process_numerical_data(df, target, standardise=False)  # True for standardising numerical data; False for normalising numerical data
df = encode_non_numerical_data(df, target)

'''
# Executing all feature selection techniques
os.environ['min_features'] = str(math.floor(math.sqrt(len(df.columns) - 1)))
best_features, models = execute_fs(df, target)

with open(fs_file, 'a') as f:
  f.write(f"Base features: {len(df.columns) - 1}\n")
for m, model in enumerate(models):
  string = model[:-1] + f"({len(best_features[m]) - 1}): "
  string += str(best_features[m]) + "\n"
  with open(fs_file, 'a') as f:
    f.write(string)
'''

best_features = []
with open(fs_file, 'r') as file:
  lines = file.readlines()[1:11]
  for line in lines:
    start_idx = line.find('[')
    end_idx = line.find(']')
    if start_idx != -1 and end_idx != -1:
      string_list = line[start_idx:end_idx+1]
      best_features.append(eval(string_list))

models = ["", "GA-", "RF-", "RFE-", "RFECV-", "GWO-", "WOA-", "HHO-", "FA-", "CS-", "BA-"]

# Applying the feature selection to dataframe
dfs = [df]
for i in range(len(best_features)):
  dfs.append(df[best_features[i]])


def get_memory_usage():
  process = psutil.Process(os.getpid())
  mem_info = process.memory_info()
  memory_usage_in_gb = mem_info.rss / (1024 ** 3)
  return memory_usage_in_gb

i_s = [1, 5, 6, 9]

# Ensemble
for i, d in enumerate(dfs):
  if i not in i_s:
    continue
  print("Current time:", datetime.datetime.now())
  keras.utils.set_random_seed(42)  # 1, 15, 42
  tf.config.experimental.enable_op_determinism()
  tf.config.run_functions_eagerly(True)
  tf.data.experimental.enable_debug_mode()

  model_name = models[i] + "Ensemble"
  model = f'Model: {model_name}'
  print(model)
  # print(d.columns)
  start_time = time.time()
  initial_memory = get_memory_usage()

  results = ensemble(d, target)

  end_time = time.time()
  final_memory = get_memory_usage()

  duration = int(np.round((end_time - start_time) / 60))
  total_memory_used = final_memory - initial_memory

  content = "\n" + model + f" ({duration})" + "\n" + str(results)
  content += "\n" + f"Memory usage: {total_memory_used:.2f} GB"
  with open(fs_file, 'a') as f:
    f.write(content)
'''


# WOCLSA
opts = {
    'N': 10,       # Number of whales
    'T': 3,        # Maximum number of iterations
    'b': 1         # Constant
}

for i, d in enumerate(dfs):
  print("Current time:", datetime.datetime.now())
  keras.utils.set_random_seed(1)  # 1, 15, 42
  tf.config.experimental.enable_op_determinism()
  tf.config.run_functions_eagerly(True)
  tf.data.experimental.enable_debug_mode()

  model_name = models[i] + "WOCLSA"
  model = f"Model: {model_name}"
  print(model)
  # print("Features:", d.columns)
  start_time = time.time()
  initial_memory = get_memory_usage()

  results = woclsa(d, target, opts)

  end_time = time.time()
  final_memory = get_memory_usage()
  
  duration = int(np.round((end_time - start_time) / 60))
  total_memory_used = final_memory - initial_memory

  content = "\n" + model + f" ({duration})" + "\n" + str(results)
  content += "\n" + f"Memory usage: {total_memory_used:.2f} GB"
  with open(fs_file, 'a') as f:
    f.write(content)
'''
