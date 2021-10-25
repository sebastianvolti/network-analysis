from constants import atlas_switcher, correlation_switcher

# GENERAL IMPORTS
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx 
import stellargraph as sg
import pandas as pd
import sys
import random


#load rois files from drive
def load_roi(file_name):
  roi = loadmat(file_name)
  data = [[row for row in line] for line in roi['ROISignals']]
  df_rois = pd.DataFrame(data)
  return df_rois

#load extra info por subjects
def load_extra_info():
  extra_info = pd.read_csv('extra-info/info.csv')  
  extra_info[['Age', 'Sex']] = extra_info[['Age', 'Sex']].apply(lambda x:  round((x - x.min()) / (x.max() - x.min()), 5))
  extra_info_map = {}
  for i , row in extra_info.iterrows():
    subject_id = extra_info.at[i,'ID']
    sex = extra_info.at[i,'Sex']
    age = extra_info.at[i,'Age']
    extra_info_map[subject_id] = {"sex": sex, "age": age}
  return extra_info_map

#extract global signal from roi dataframe
def select_global_signal(df_rois):
  selected_atlas = atlas_switcher.get(7, "Invalid Atlas")
  atlas_start_range = selected_atlas["start_range"]
  atlas_end_range = selected_atlas["end_range"]
  atlas_range = range(atlas_start_range, atlas_end_range)
  df_atlas = df_rois[atlas_range]
  return df_atlas

#extract atlas from roi dataframe
def select_atlas(df_rois, atlas_index):
  if (atlas_index == 9):
      rows = select_random_rois(df_rois, 150)
      df_atlas = df_rois[rows]
  else:
      selected_atlas = atlas_switcher.get(atlas_index, "Invalid Atlas")
      atlas_start_range = selected_atlas["start_range"]
      atlas_end_range = selected_atlas["end_range"]
      atlas_range = range(atlas_start_range, atlas_end_range)
      df_atlas = df_rois[atlas_range]
      
  return df_atlas


def select_random_rois(df_rois, rois_quantity):
  rois = [*range(0, 1833, 1)]
  rows = []
  for index in range(rois_quantity):
    elem = random.choice(rois)
    rois.remove(elem)
    rows.append(elem)

  return rows


#plot correlation matrix
def plot_correlation_matrix(correlation_matrix):
  plt.imshow(correlation_matrix)
  plt.colorbar()
  plt.show()


def calculate_pearson_correlation(df_atlas):
  return np.corrcoef(df_atlas.T)


def calculate_pearson_fisher_correlation(df_atlas):
  corr = np.corrcoef(df_atlas.T)
  fisher = np.arctanh(corr)

  return fisher


def partial_corr(p1,p2,p3):
    p1k = np.expand_dims(p1, axis=1)
    p2k = np.expand_dims(p2, axis=1)
    p3k = np.expand_dims(p3, axis=1)
    
    beta_p2k = linalg.lstsq(p1k, p2k)[0]  # Calculating Weights (W*)
    beta_p3k = linalg.lstsq(p1k, p3k)[0]  # Calculating Weights(W*)
    res_p2k = p2k - p1k.dot(beta_p2k) # Calculating Errors
    res_p3k = p3k - p1k.dot(beta_p3k) # Calculating Errors
    corr = stats.pearsonr(np.squeeze(res_p2k), np.squeeze(res_p3k))[0]

    return corr


def convert_matrix_to_igraph(correlation_matrix):
  g = ig.Graph.Weighted_Adjacency(correlation_matrix)
  edges = []
  for elem in g.es:
    if (elem['weight'] > min_edge_weight and elem['weight'] != 1):
      edges.append(elem)

  return g.subgraph_edges(edges)


def convert_matrix_to_networkx(correlation_matrix, correlation, thresh, binarize_coef):
  if (correlation == "Pearson Correlation"):
    aux_g = nx.from_numpy_matrix(correlation_matrix)
    g = aux_g.copy()
    for u,v,a in aux_g.edges(data=True):
      if (float(a['weight']) == 1):
        g.remove_edge(u, v)

      if (float(a['weight']) < binarize_coef):
        g.remove_edge(u, v)
    return g
  elif (correlation == "Pearson Correlation and Fisher Normalization"):
    correlation_matrix[correlation_matrix > thresh]=1
    correlation_matrix[correlation_matrix <= thresh]=0
    g = nx.from_numpy_matrix(correlation_matrix)
    return g
  else:
    #Partial Correlation
    g = nx.from_numpy_matrix(correlation_matrix)
    return g


def generate_igraph_plot(graph):
  return ig.plot(graph, bbox=(0, 0, 500, 500))


def nodal_eff(g):
    weights = g.es["weight"][:]
    sp = (1.0 / np.array(g.shortest_paths_dijkstra(weights=weights)))
    np.fill_diagonal(sp,0)
    N=sp.shape[0]
    ne= (1.0/(N-1)) * np.apply_along_axis(sum,0,sp)

    return ne


#Generate example instance for each ROI in path folder
def generate_examples_list(folder_path, atlas_id, correlation, thresh, binarize_coef, extra_info = {}):
  example_list = []
  file_list = os.listdir(folder_path)
  file_list_len = len(file_list)
  for ind, file_name in enumerate(file_list):
    if (file_name != ".DS_Store" and file_name != "DS_Store"):
      dt = file_name.split('_')[1].split(".")[0].split('-')
      dt[0] = dt[0][1:] 
      file_path = folder_path + file_name
      df_rois = load_roi(file_path)
      df_atlas = select_atlas(df_rois, atlas_id)
      global_signal = 0
      global_signal_avg = 0
      #global_signal = select_global_signal(df_rois)
      #global_signal_avg = np.average(global_signal)
      selected_correlation = correlation_switcher.get(correlation, "Invalid Correlation")
      if (correlation == "Pearson Correlation"):
          correlation_matrix = calculate_pearson_correlation(df_atlas)
      elif (correlation == "Pearson Correlation and Fisher Normalization"):
          correlation_matrix = calculate_pearson_fisher_correlation(df_atlas)
      else:
          #Partial Correlation
          correlation_matrix = {}
      graph = convert_matrix_to_networkx(correlation_matrix, correlation, thresh, binarize_coef)
      experiment = {"id": dt[2], "class": dt[1], 'exp': dt[0], 'graph': graph, 'gs': global_signal_avg}
      
      if (extra_info) :
        subject_id = file_name.split('_')[1].split(".")[0]
        experiment["age"] = extra_info[subject_id]["age"]
        experiment["sex"] = extra_info[subject_id]["sex"]

      example_list.append(experiment)
  return example_list