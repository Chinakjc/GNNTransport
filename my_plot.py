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
from matplotlib.collections import LineCollection 

def plot_real_vs_pred(y_pred,y_real):
    # Convert to NumPy array
    predict_graphs_nor = y_pred
    actual_labels_nor = y_real

    fig = plt.figure(figsize=(8, 6))  

    # Draw a dashed line for y=x, and add a legend label. 
    plt.plot([0, 1], [0, 1],  
            linestyle='--',        # Dashed line  
            color='green',  
            linewidth=2,  
            label='y = x')  

    #index_i = random.sample(range(0, len(predict_graphs_nor[0])), 300)
    # If there are multiple datasets, they can be marked separately.
    for j in range(len(predict_graphs_nor)):  
        x = actual_labels_nor[j]  
        y = predict_graphs_nor[j]   
        plt.scatter(  
            x, y,  
            c='C{}'.format(j),     # Automatic cycling color scheme C0, C1, ...
            s=30,  
            alpha=0.7,  
            label=f'Prediction set {j+1}'  
        )  

    plt.xlabel('Real Traffic Flow', fontsize=14)  
    plt.ylabel('Predicted Traffic Flow', fontsize=14)  
    plt.title('Real vs Predicted Scatter Plot', fontsize=16)  
    plt.grid(True, linestyle=':', alpha=0.5)  
    plt.legend(loc='upper left')    # Put the legend in the upper left corner. 
    plt.xlim(0, 1)  
    plt.ylim(0, 1)  

    plt.tight_layout()  
    plt.show()

def plot_real_vs_pred_subsample(y_pred,y_real,n_samples):

    predict_graphs_nor = y_pred
    actual_labels_nor = y_real
    index_i = random.sample(range(0, len(predict_graphs_nor[0])), n_samples)
   

    fig = plt.figure(figsize=(8, 6))  

    # Draw a dashed line for y=x, and add a legend label. 
    plt.plot([0, 1], [0, 1],  
            linestyle='--',        # Dashed line   
            color='green',  
            linewidth=2,  
            label='y = x')  

    # If there are multiple datasets, they can be marked separately.
    for j in range(len(predict_graphs_nor)):  
        x = actual_labels_nor[j][index_i]  
        y = predict_graphs_nor[j][index_i]    
        plt.scatter(  
            x, y,  
            c='C{}'.format(j),     # Automatic cycling color scheme C0, C1, ...
            s=30,  
            alpha=0.7,  
            label=f'Prediction set {j+1}'  
        )  

    plt.xlabel('Real Traffic Flow', fontsize=14)  
    plt.ylabel('Predicted Traffic Flow', fontsize=14)  
    plt.title('Real vs Predicted Scatter Plot', fontsize=16)  
    plt.grid(True, linestyle=':', alpha=0.5)  
    plt.legend(loc='upper left')    #  Put the legend in the upper left corner. 
    plt.xlim(0, 1)  
    plt.ylim(0, 1)  

    plt.tight_layout()  
    plt.show() 

def parse_network(xml_path: str):  
    """  
    Parse MATSim network XML, remove namespace (if any),
    Return:
    nodes      : dict[node_id] = (x, y)
    link_ids   : [id1, id2, ...]
    link_lines : [((x1,y1),(x2,y2)), ...]
    bbox       : (min_x, max_x, min_y, max_y) 
    """  
    tree = ET.parse(xml_path)  
    root = tree.getroot()  
    # remove namespace  
    for e in root.iter():  
        if '}' in e.tag:  
            e.tag = e.tag.split('}', 1)[1]  

    # noes  
    nodes = {}  
    xs, ys = [], []  
    for n in root.find('nodes').findall('node'):  
        nid = n.attrib['id']  
        x, y = float(n.attrib['x']), float(n.attrib['y'])  
        nodes[nid] = (x, y)  
        xs.append(x); ys.append(y)  

    # links  
    link_elems = root.find('links').findall('link')  
    link_ids   = [lk.attrib['id'] for lk in link_elems]  
    link_lines = [(nodes[lk.attrib['from']], nodes[lk.attrib['to']])  
                  for lk in link_elems]  

    # Global boundary
    min_x, max_x = min(xs), max(xs)  
    min_y, max_y = min(ys), max(ys)  
    return nodes, link_ids, link_lines, (min_x, max_x, min_y, max_y)  

def load_policy_links(policy_links_txt: str):  
    """  
    Read a list of policy link IDs and return a set.
    """  
    with open(policy_links_txt) as f:  
        return {ln.strip() for ln in f if ln.strip()}  

def plot_policy_network_2panels(  
    net_xml_path: str,  
    policy_links_txt: str,  
    hrs_no_policy: list,  
    hrs_pred: list,  
    hrs_real: list,  
    cmap: str = 'RdYlBu_r',
    c_title=''
):  
    eps = 10e-10
    clip_pct = 500
    # 1. Load network structure & policy link 
    nodes, link_ids, link_lines, bbox = parse_network(net_xml_path)  
    policy_set = load_policy_links(policy_links_txt)  

    # Black highlighted section  
    black_lines = [link_lines[i]  
                   for i,lid in enumerate(link_ids)  
                   if lid in policy_set]  

    #2. Calculating Percentage Change
    hrs_no   = np.array(hrs_no_policy, dtype=float)  
    pct_pred = (np.array(hrs_pred, dtype=float) - hrs_no) / (hrs_no+eps) * 100  
    pct_real = (np.array(hrs_real, dtype=float) - hrs_no) / (hrs_no+eps) * 100  
    pct_pred = np.clip(pct_pred, -clip_pct, clip_pct)  
    pct_real = np.clip(pct_real, -clip_pct, clip_pct)  
    #print(pct_real)

    vmin = min(pct_pred.min(), pct_real.min())  
    vmax = max(pct_pred.max(), pct_real.max())  
    norm     = plt.Normalize(vmin=vmin, vmax=vmax)  
    cmap_obj = plt.get_cmap(cmap)  

    # 3. Draw a two-panel plot 
    min_x, max_x, min_y, max_y = bbox  
    pad_x = (max_x - min_x) * 0.03  
    pad_y = (max_y - min_y) * 0.03  

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  
    for ax, pct, title in zip(  
        axes,  
        (pct_pred, pct_real),  
        ('Predicted % change', 'Actual % change')  
    ):  
        # Background gray mesh  
        ax.add_collection(LineCollection(  
            link_lines, colors='lightgrey', linewidths=1.5, zorder=1))  
        # Policy dark web 
        ax.add_collection(LineCollection(  
            black_lines, colors='black', linewidths=2.5, zorder=2))  
        # Heatmap  
        colors = cmap_obj(norm(pct))  
        ax.add_collection(LineCollection(  
            link_lines, colors=colors, linewidths=1.0, zorder=3))  

        ax.set_title(title)  
        ax.set_aspect('equal')  
        ax.set_xticks([]); ax.set_yticks([])  
        ax.set_xlim(min_x - pad_x, max_x + pad_x)  
        ax.set_ylim(min_y - pad_y, max_y + pad_y)  

    # 4. Vertical colorbar on the right side 
    fig.subplots_adjust(right=0.85)  
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]  
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)  
    sm.set_array([])  
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')  
    cbar.set_label('Change in % of traffic flow' + c_title)  

    plt.show() 