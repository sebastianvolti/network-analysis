import time
import math
import numpy as np
import pandas as pd
import networkx as nx 
import networkx.algorithms.community as nx_comm
from sklearn import svm, naive_bayes, neighbors, metrics
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
from mrmr import mrmr_classif
from keras import backend as K
import matplotlib.pyplot as plt

import random


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def stat(name, y, y_pred, n=0):
    print("{}-{}".format(name, n), end='')
    print("\tAcc:{} ".format(metrics.accuracy_score(y, y_pred)), end='')
    print("\tPrc:{} ".format(metrics.precision_score(y, y_pred)), end='')
    print("\tRec:{} ".format(metrics.recall_score(y, y_pred)), end='')
    print("\tF1s:{} ".format(metrics.f1_score(y, y_pred)))
    return metrics.accuracy_score(y, y_pred), metrics.precision_score(y, y_pred), metrics.recall_score(y, y_pred), metrics.f1_score(y, y_pred)


def model_exec(name, X_tr, y_tr, X_te, y_te, model, results_map):
    stat_pipe = []
    start = time.time()

    feature_names = ["ge", "le", "mod", "bc_avg", "cs_avg", "dg_avg", "clq_avg", "trn_avg", "sq_cs_avg","sex", "age"]
    model.fit(X_tr, y_tr)
    std = []

    #plot feature importances for random forest
    
    #importances = model.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    #forest_importances = pd.Series(importances, index=feature_names)
    #fig, ax = plt.subplots()
    #forest_importances.plot.bar(yerr=std, ax=ax)
    #ax.set_title("Feature importances")
    #ax.set_ylabel("Mean decrease in impurity")
    #fig.savefig(figure_name_img + "test", bbox_inches='tight', pad_inches=0)

    middle = time.time()
    y_pred = model.predict(X_te)
    for index in range(len(y_pred)): 
      results_map[index]["predictions"].append(y_pred[index])
    stat_pipe.append((name, y_te, y_pred))
    acc, pre, rec, f1 = stat(name, y_te, y_pred)
    end = time.time()

   

    print("{}: tr {:.4f}s, te {:.4f}s".format(name,middle-start, end-middle))
    return acc, pre, rec, f1, results_map

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

          degrees = nx.degree(exp["graph"])
          degree_values = {k: v for k, v in degrees}
          cliques = nx.node_clique_number(exp["graph"])
          triangles = nx.triangles(exp["graph"])
          sq_cs = nx.square_clustering(exp["graph"])

          exp["metrics"]["dg_avg"] = get_safe_value(round(np.average(list(degree_values.values())), 5))
          exp["metrics"]["clq_avg"] = get_safe_value(round(np.average(list(cliques.values())), 5))
          exp["metrics"]["trn_avg"] = get_safe_value(round(np.average(list(triangles.values())), 5))
          exp["metrics"]["sq_cs_avg"] = get_safe_value(round(np.average(list(sq_cs.values())), 5))
        


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
  random.seed(10)
  random.shuffle(exp_list)
  file_list_len = len(exp_list)
  for ind, exp in enumerate(exp_list):
      print("\r{}/{}/{:2f}s".format(ind, file_list_len, time.time()-ta), end='', flush=True)
      ta = time.time()

      graph = exp["graph"]
      bc = nx.betweenness_centrality(graph)
      cs = nx.clustering(graph)
      degrees = nx.degree(graph)
      degree_values = {k: v for k, v in degrees}
      cliques = nx.node_clique_number(graph)
      triangles = nx.triangles(graph)
      sq_cs = nx.square_clustering(graph)
    
      for node in graph.nodes():
        graph.nodes[node]['feature'] = [bc[node], cs[node], sq_cs[node], degree_values[node], cliques[node], triangles[node]]

      exp["graph"] = graph

  return exp_list

#Select Best Features with MRMR
def select_best_features(X, X_val, y, y_val, quantity):

  X_df = pd.DataFrame(X)
  y_df = pd.Series(y)

  # use mrmr classification
  selected_features = mrmr_classif(X_df, y_df, K = quantity)

  X_selected = []
  for features in X:
      new_features = [features[i] for i in selected_features]
      X_selected.append(new_features) 
  
  X_val_selected = []
  for features in X_val:
      new_features = [features[i] for i in selected_features]
      X_val_selected.append(new_features) 

  return X_selected, X_val_selected

#Separate Training and Validation datasets
def separate_training_validation(exp_list, feature_selection_id):  
  X = []
  X_val = []
  y = []
  y_val = []
  clinica_train = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 0,
    22: 0,
    23: 0,
    24: 0,
    25: 0,
    26: 0,
  }
  clinica_val = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 0,
    22: 0,
    23: 0,
    24: 0,
    25: 0,
    26: 0,
  }
  pos_neg = {
    0: 0,
    1: 0
  }
  random.seed(10)
  random.shuffle(exp_list)
  ta = time.time()
  file_list_len = len(exp_list)
  val_init = int(len(exp_list)*0.8)
  exp_to_validate = range(val_init, file_list_len)

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

      dg_avg = exp["metrics"]["dg_avg"]
      clq_avg = exp["metrics"]["clq_avg"]
      trn_avg = exp["metrics"]["trn_avg"]
      sq_cs_avg = exp["metrics"]["sq_cs_avg"]

      cls = exp['class']
      

      
      #default features, feature_selection_id = 0
      feature_list = [ge, le, mod, bc_avg, cs_avg]
        
      if (feature_selection_id == 1): 
        feature_list.append(gs_avg)
      elif (feature_selection_id == 2): 
        feature_list.append(sex)
        feature_list.append(age)
      elif (feature_selection_id == 3):
        feature_list.append(dg_avg)
        feature_list.append(clq_avg)
        feature_list.append(trn_avg)
        feature_list.append(sq_cs_avg)
        feature_list.append(sex)
        feature_list.append(age)


      if int(ind) in exp_to_validate:
          X_val.append(feature_list)
          y_val.append(int(cls))
  
          key = int(exp["exp"])
          clinica_val[key]+=1

          key_pos_neg = int(cls)
          pos_neg[key_pos_neg]+=1

      else:
          X.append(feature_list)
          y.append(int(cls))
          key = int(exp["exp"])
          clinica_train[key]+=1

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
      graphs_labels.append(1 if g["class"] == 1 else 0)

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
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["acc",f1_m,precision_m, recall_m])

    return model, generator

def train_fold(model, train_gen, test_gen, epochs):
    es = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
    )
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
    )
    # calculate performance on the test data and return along with history
    loss, accuracy, f1_score, precision, recall = model.evaluate(test_gen, verbose=0)

    return history, accuracy

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
            train_index, test_index, graph_labels, generator, batch_size=4
        )
        model, generator = create_graph_classification_model(generator)
        history, acc = train_fold(model, train_gen, test_gen, epochs)
        test_accs.append(acc)

    print(
        f"Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%"
    )

    return model, generator, test_accs
  
#Execute GCN model
def execute_gcn_model(model, generator, len_exp_list):
    val_init = int(len_exp_list*0.8)
    val_range = len_exp_list - val_init
    exp_to_validate = range(val_init, len_exp_list)
    pred = model.predict(generator.flow(exp_to_validate))
    print(pred)


#Execute LR model
def execute_logreg_model(X, y, X_val, y_val, results_map):  
  return model_exec("Log Reg",X, y, X_val, y_val, LogisticRegression(max_iter = 500), results_map)

def execute_rf_model(X, y, X_val, y_val, results_map):  
  return model_exec("Random Forest",X, y, X_val, y_val, RandomForestClassifier(), results_map)

#Execute SVM model
def execute_svm_model(X, y, X_val, y_val, results_map):  
  hyperparams = {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'}
  svm_model = svm.SVC(C=hyperparams['C'], gamma=hyperparams['gamma'], kernel=hyperparams['kernel'])
  return model_exec("SVM Class",X, y, X_val, y_val, svm_model, results_map)

#Execute SVM Linear model
def execute_svm_linear_model(X, y, X_val, y_val, results_map):  
  svm_model = svm.LinearSVC()
  return model_exec("SVM Linear Class",X, y, X_val, y_val, svm_model, results_map)

#Execute KNN model
def execute_knn_model(X, y, X_val, y_val, results_map):  
  knn_model = neighbors.KNeighborsClassifier()
  return model_exec("KNN Class",X, y, X_val, y_val, knn_model, results_map)

#Execute Naive Bayes model
def execute_nb_model(X, y, X_val, y_val, results_map):  
  nb_model = naive_bayes.GaussianNB()
  return model_exec("NB Class",X, y, X_val, y_val, nb_model, results_map)


def init_results_map(y_val):
  results_map = {}
  for index in range(len(y_val)):
    results_map[index] = {
        "real_tag": y_val[index],
        "predictions": []
    }

  return results_map

def process_results(results_map):
  contador_uno = 0
  contador_cero = 0
  for index in range(len(results_map)):
    for pred in range(3):
      value = results_map[index]["predictions"][pred]
      if (value == 0):
        contador_cero+=1
      else:
        contador_uno+=1

      
def process_results_map(results_map, name, y_val):
  y_pred = []
  for index in range(len(results_map)):
    results_average = np.average(results_map[index]["predictions"])
    if (results_average > 0.5):
      results_map[index]["average"] = 1
    else:
      results_map[index]["average"] = 0  
    y_pred.append(results_map[index]["average"])
  
  return calculate_final_results(name, y_val, y_pred)

def calculate_final_results(name, y_val, y_pred):
    stat_pipe = []
    results = stat(name, y_val, y_pred)

    return results
      