import time
import math
import numpy as np
import pandas as pd
import networkx as nx 
import networkx.algorithms.community as nx_comm
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
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
          exp["metrics"]["gs_avg"] = round(exp["gs"], 5)
          try:
              c = nx_comm.greedy_modularity_communities(exp["graph"])
              exp["metrics"]["mod"] = get_safe_value(round(nx_comm.modularity(exp["graph"], c), 5))
          except KeyError:
              print("\n {}/{}/{}".format(ind, exp['id'], exp["exp"]))
              exp["metrics"]["mod"] = 0
  return exp_list

#Statistic Analisys
def calculate_statistical_features_gcn(exp_list):
  ta = time.time()
  file_list_len = len(exp_list)
  for ind, exp in enumerate(exp_list):
      print("\r{}/{}/{:2f}s".format(ind, file_list_len, time.time()-ta), end='', flush=True)
      ta = time.time()

      graph = exp["graph"]
      bc = nx.betweenness_centrality(graph)
      cs = nx.clustering(graph)
      degrees = nx.degree(graph)
      degree_values = {k: v for k, v in degrees}

      for node in graph.nodes():
        graph.nodes[node]['feature'] = [bc[node], cs[node], degree_values[node]]

      exp["graph"] = graph

  return exp_list

#Separate Training and Validation datasets
def separate_training_validation(exp_list, feature_selection_id):  
  X = []
  X_val = []
  y = []
  y_val = []
  random.seed(10)
  random.shuffle(exp_list)
  ta = time.time()
  file_list_len = len(exp_list)
  #exp_to_validate = range(118, 148)
  exp_to_validate = range(15, 20)
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
      sex = exp["sex"]
      age = exp["age"]
      cls = exp['class'] #1 if exp['class']=="1" else -1
      
      #default features, feature_selection_id = 0
      feature_list = [ge, le, mod, bc_avg, cs_avg]
        
      if (feature_selection_id == 1): 
        feature_list.append(gs_avg)
      elif (feature_selection_id == 2): 
        feature_list.append(sex)
        feature_list.append(age)
      elif (feature_selection_id == 3):
        feature_list.append(gs_avg)
        feature_list.append(sex)
        feature_list.append(age)

      if int(ind) in exp_to_validate:
          #X_val.append([ge, le, mod, bc, cs, bc_avg, cs_avg, gs_avg])
          X_val.append(feature_list)
          y_val.append(int(cls))
      else:
          X.append(feature_list)
          y.append(int(cls))

  return X, X_val, y, y_val

#Separate for cross validation
def separate_cross_validation(exp_list, feature_selection_id):  
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
      gs_avg = exp["metrics"]["gs_avg"]
      sex = exp["sex"]
      age = exp["age"]
      cls = exp['class'] #1 if exp['class']=="1" else -1

      #default features, feature_selection_id = 0
      feature_list = [ge, le, mod, bc_avg, cs_avg]
        
      if (feature_selection_id == 1): 
        feature_list.append(gs_avg)
      elif (feature_selection_id == 2): 
        feature_list.append(sex)
        feature_list.append(age)
      elif (feature_selection_id == 3):
        feature_list.append(gs_avg)
        feature_list.append(sex)
        feature_list.append(age)

      train.append(feature_list)
      label.append(int(cls))
          
  return train, label

def get_safe_value(value):
  if (math.isnan(value)):
    return 0
  else:
    return value

def generate_stellar_graph(exp_list):
  graphs = []
  graphs_labels = []
  ta = time.time()
  for ind, g in enumerate(exp_list):
      print("\r{}/{}/{:2f}s".format(ind, len(exp_list), time.time()-ta), end='', flush=True)
      sg_g = sg.StellarGraph.from_networkx(g['graph'], node_features="feature")
      graphs.append(sg_g)
      graphs_labels.append(1 if g["class"] == "1" else -1)
  
  graph_labels = pd.Series(graphs_labels, name="label")
  graph_labels = pd.get_dummies(graph_labels, drop_first=True)
  generator = PaddedGraphGenerator(graphs=graphs)

  return graphs, graph_labels, generator

def create_graph_classification_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["acc"])

    return model, generator

def train_fold(model, train_gen, test_gen, epochs):
    es = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
    )
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
    )
    # calculate performance on the test data and return along with history
    test_metrics = model.evaluate(test_gen, verbose=0)
    test_acc = test_metrics[model.metrics_names.index("acc")]

    return history, test_acc

def get_generators(train_index, test_index, graph_labels, generator, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )

    return train_gen, test_gen

def train_gcn_model(folds, n_repeats, graph_labels, epochs, generator):
    test_accs = []
    stratified_folds = model_selection.RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=n_repeats
    ).split(graph_labels, graph_labels)
    for i, (train_index, test_index) in enumerate(stratified_folds):
        print(f"Training and evaluating on fold {i+1} out of {folds * n_repeats}...")
        train_gen, test_gen = get_generators(
            train_index, test_index, graph_labels, generator, batch_size=16
        )
        model, generator = create_graph_classification_model(generator)
        history, acc = train_fold(model, train_gen, test_gen, epochs)
        test_accs.append(acc)

    print(
        f"Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%"
    )

    return model, generator
  
#Execute GCN model
def execute_gcn_model(model, generator):
    pred = model.predict(generator.flow(range(10)))
    np.count_nonzero(pred != pred[0])
    print(pred)

#Execute LR model
def execute_logreg_model(X, y, X_val, y_val):  
  return model_exec("Log Reg",X, y, X_val, y_val, LogisticRegression(max_iter = 500))

#Execute SVM model
def execute_svm_model(X, y, X_val, y_val):  
  hyperparams = {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'}
  svm_model = svm.SVC(C=hyperparams['C'], gamma=hyperparams['gamma'], kernel=hyperparams['kernel'])
  return model_exec("SVM Class",X, y, X_val, y_val, svm_model)
