import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # For TF2.16+.
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn
import gzip  
import xml.etree.ElementTree as ET  
from graphviz import Digraph  
from pathlib import Path  
from typing import Tuple , Optional, List
import os  
import re  
import glob  
import tensorflow as tf  
import random
import json  
import time

# Function to build a regression model
def build_regression_model(
    graph_tensor_spec,
    node_dim=64,
    edge_dim=16,
    message_dim=64,
    next_state_dim=64,
    output_dim=25,  # Regression outputs for 24 H and daily HRS avg
    num_message_passing=3,
    l2_regularization=5e-4,
    dropout_rate=0.2,
    hiden_dim = 64,

):
    input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    graph = input_graph.merge_batch_to_components()

    # Set initial node state
    def set_initial_node_state(node_set, *, node_set_name):
        try:  
            feat_tensors = list(node_set.features.values())  
            #feat_tensors = [tf.keras.layers.Flatten(k) for k in feat_tensors] 
            #print("yes")
        except AttributeError:  
            # Method B: Treat node_set as a Mapping, using keys()/__getitem__ 
            feat_tensors = [node_set[k] for k in node_set.keys()]  
            #print("no")

         # 2) Concatenated into [num_nodes, total_feat_dim] 
        features = tf.concat(feat_tensors, axis=-1)  
        print("f = ",features)

        features = tf.keras.layers.Dense(node_dim, activation="relu")(features)
        features = tf.keras.layers.Dense(node_dim, activation="relu")(features)
        features = tf.keras.layers.Dense(node_dim, activation="relu")(features)
     
        return features

         

    def set_initial_edge_state(edge_set, *, edge_set_name):  
        try:  
            feat_tensors = list(edge_set.features.values()) 
            #feat_tensors = [tf.reshape(k,[-1]) for k in feat_tensors] 
            #feat_tensors = [tf.keras.layers.Flatten(k) for k in feat_tensors] 
        except AttributeError:  
            # Method B: Treat edge_set as a Mapping, using keys()/__getitem__.
            feat_tensors = [edge_set[k] for k in edge_set.keys()]  
        features = tf.concat(feat_tensors, axis=-1)
        features = tf.keras.layers.Dense(edge_dim, activation="relu")(features)
        return features

    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state
    )(graph)

    # Define helper function for Dense layers with regularization and Dropout.
    def dense(units, activation="relu"):
        regularizer = tf.keras.regularizers.l2(l2_regularization)
        return tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=regularizer, bias_regularizer=regularizer),
            tf.keras.layers.Dropout(dropout_rate)
        ])

    # GNN core, performing multiple rounds of message passing.
    for i in range(num_message_passing):
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "links": tfgnn.keras.layers.NodeSetUpdate(
                    {
                        "line_graph": tfgnn.keras.layers.SimpleConv(
                            message_fn=dense(message_dim),
                            reduce_type="sum",
                            receiver_tag=tfgnn.TARGET
                        )
                    },
                    tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim))
                )
            }
        )(graph)


    node_states = tfgnn.keras.layers.Readout(node_set_name="links")(graph) #[batch_size*num_nodes,next_state_dim]


    #print(node_states)
    #print("shape=",node_states.shape[0])
    res = tf.expand_dims(node_states, axis=1) # [batch_size*num_nodes, 1, next_state_dim]
    def unflatten(x):
        return tf.reshape(x,[-1,node_states.shape[0],next_state_dim]) # [batch_size, num_nodes, next_state_dim]
    res = tf.keras.layers.Lambda(unflatten)(res)
    res = tf.keras.layers.Dense(hiden_dim,activation='relu')(res)
    res = tf.keras.layers.Dense(hiden_dim,activation='relu')(res)
    res = tf.keras.layers.Dense(hiden_dim,activation='relu')(res)
    res = tf.keras.layers.Dense(output_dim)(res)


    return tf.keras.Model(inputs=[input_graph], outputs=[res])


 