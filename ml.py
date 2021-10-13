import time
import math
import numpy as np
import networkx as nx 
import networkx.algorithms.community as nx_comm
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import random

def stat(name, y, y_pred, n=0):
    print("{}-{}".format(name, n), end='')
    print("\tAcc:{} ".format(metrics.accuracy_score(y, y_pred)), end='')
    print("\tPrc:{} ".format(metrics.precision_score(y, y_pred)), end='')
    print("\tRec:{} ".format(metrics.recall_score(y, y_pred)), end='')
    print("\tF1s:{} ".format(metrics.f1_score(y, y_pred)))
    return metrics.accuracy_score(y, y_pred), metrics.precision_score(y, y_pred), metrics.recall_score(y, y_pred), metrics.f1_score(y, y_pred)


def model_exec(name, X_tr, y_tr, X_te, y_te, model):
    stat_pipe = []
    start = time.time()
    model.fit(X_tr, y_tr)
    middle = time.time()
    y_pred = model.predict(X_te)
    stat_pipe.append((name, y_te, y_pred))
    acc, pre, rec, f1 = stat(name, y_te, y_pred)
    end = time.time()
    print("{}: tr {:.4f}s, te {:.4f}s".format(name,middle-start, end-middle))
    return acc, pre, rec, f1

#Statistic Analisys
def calculate_statistical_features(exp_list):
  ta = time.time()
  file_list_len = len(exp_list)
  for ind, exp in enumerate(exp_list):
      print("\r{}/{}/{:2f}s".format(ind, file_list_len, time.time()-ta), end='', flush=True)
      ta = time.time()
      if "metrics" not in exp or \
          "mod" not in exp["metrics"] or\
          exp["metrics"]["mod"] is None:
          exp["metrics"] = {}
          exp["metrics"]["ge"] = get_safe_value(round(nx.global_efficiency(exp["graph"]), 5))
          exp["metrics"]["le"] = get_safe_value(round(nx.local_efficiency(exp["graph"]), 5))
          bc_values = nx.betweenness_centrality(exp["graph"])
          exp["metrics"]["bc"] = get_safe_value(max(bc_values, key=bc_values.get))
          exp["metrics"]["bc_avg"] = get_safe_value(round(np.average(list(bc_values.values())), 5))
          cs_values = nx.clustering(exp["graph"])
          exp["metrics"]["cs"] =  get_safe_value(max(cs_values, key=cs_values.get))
          exp["metrics"]["cs_avg"] = get_safe_value(round(np.average(list(cs_values.values())), 5))
          exp["metrics"]["gs_avg"] = exp["gs"]
          try:
              c = nx_comm.greedy_modularity_communities(exp["graph"])
              exp["metrics"]["mod"] = get_safe_value(round(nx_comm.modularity(exp["graph"], c), 5))
          except KeyError:
              print("\n {}/{}/{}".format(ind, exp['id'], exp["exp"]))
              exp["metrics"]["mod"] = 0
  return exp_list


#Separate Training and Validation datasets
def separate_training_validation(exp_list):  
  X = []
  X_val = []
  y = []
  y_val = []
  random.seed(10)
  random.shuffle(exp_list)
  ta = time.time()
  file_list_len = len(exp_list)
  exp_to_validate = range(118, 148)
  for ind, exp in enumerate(exp_list):
      print("\r{}/{}/{:2f}s".format(ind, file_list_len, time.time()-ta), end='', flush=True)
      ge = exp["metrics"]["ge"]
      le = exp["metrics"]["le"]
      mod = exp["metrics"]["mod"]
      bc = exp["metrics"]["bc"]
      bc_avg = exp["metrics"]["bc_avg"]
      cs = exp["metrics"]["cs"]
      cs_avg = exp["metrics"]["cs_avg"]
      gs_avg = exp["metrics"]["gs_avg"]
      cls = exp['class'] #1 if exp['class']=="1" else -1
      if int(ind) in exp_to_validate:
          X_val.append([ge, le, mod, bc, cs, bc_avg, cs_avg, gs_avg])
          y_val.append(int(cls))
      else:
          X.append([ge, le, mod, bc, cs, bc_avg, cs_avg, gs_avg])
          y.append(int(cls))
          
  return X, X_val, y, y_val


#Separate for cross validation
def separate_cross_validation(exp_list):  
  train = []
  label = []
  random.seed(10)
  random.shuffle(exp_list)
  ta = time.time()
  file_list_len = len(exp_list)
  for ind, exp in enumerate(exp_list):
      print("\r{}/{}/{:2f}s".format(ind, file_list_len, time.time()-ta), end='', flush=True)
      ge = exp["metrics"]["ge"]
      le = exp["metrics"]["le"]
      mod = exp["metrics"]["mod"]
      bc = exp["metrics"]["bc"]
      bc_avg = exp["metrics"]["bc_avg"]
      cs = exp["metrics"]["cs"]
      cs_avg = exp["metrics"]["cs_avg"]
      cls = exp['class'] #1 if exp['class']=="1" else -1
      train.append([ge, le, mod, bc, cs, bc_avg, cs_avg])
      label.append(int(cls))
          
  return train, label


def get_safe_value(value):
  if (math.isnan(value)):
    return 0
  else:
    return value


#Execute ML models
def execute_logreg_model(X, y, X_val, y_val):  
  return model_exec("Log Reg",X, y, X_val, y_val, LogisticRegression(max_iter = 500))

#Execute ML models
def execute_svm_model(X, y, X_val, y_val):  
  hyperparams = {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'}
  svm_model = svm.SVC(C=hyperparams['C'], gamma=hyperparams['gamma'], kernel=hyperparams['kernel'])
  return model_exec("SVM Class",X, y, X_val, y_val, svm_model)
