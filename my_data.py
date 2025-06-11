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

def build_line_graph_tensor(  
    network_path: str,  
    linkstats_path: str,  
    base_linkstats_path: Optional[str] = None  
) -> Tuple[tfgnn.GraphTensor, tf.Tensor]:  
    """  
    Args:
        network_path: Path to the network file, either network.xml or network.xml.gz.
        linkstats_path: Path to linkstats.txt or linkstats.txt.gz (used for labels).
        base_linkstats_path: Path to baselinkstats.txt(.gz) (optional, used for additional features).
    Returns:
        graph: GraphTensor (line graph, nodes = links).
        labels: tf.Tensor, shape=[num_links,25], hourly HRS avg + daily average HRS avg from linkstats.
    """  

    # ————————————————  
    # 1) Parsing network, reading node coordinates and link attributes.
    opener = gzip.open if network_path.endswith('.gz') else open  
    with opener(network_path, 'rb') as f:  
        tree = ET.parse(f)  
    root = tree.getroot()  

    node_coord = {}  
    for node in root.findall('nodes/node'):  
        nid = node.attrib['id']  
        node_coord[nid] = (float(node.attrib['x']), float(node.attrib['y']))  

    link_list = []  
    for link in root.findall('links/link'):  
        lid = link.attrib['id']  
        frm, to = link.attrib['from'], link.attrib['to']  
        lf    = float(link.attrib.get('length',0.0))  
        fs    = float(link.attrib.get('freespeed',0.0))  
        cap   = float(link.attrib.get('capacity',0.0))  
        pl    = float(link.attrib.get('permlanes',0.0))  
        xf,yf = node_coord[frm]  
        xt,yt = node_coord[to]  
        link_list.append({  
            'id': lid, 'from': frm, 'to': to,  
            'length':lf,'freespeed':fs,'capacity':cap,'permlanes':pl,  
            'xf':xf,'yf':yf,'xt':xt,'yt':yt  
        })  

    link_ids = [lk['id'] for lk in link_list]  
    id2idx = {lid:i for i,lid in enumerate(link_ids)}  
    num_links = len(link_list) 

    # ————————————————  
    # 2) Read linkstats.txt and construct the labels array with dimensions [num_links, 25].
    opener = gzip.open if linkstats_path.endswith('.gz') else open  
    df_ls = pd.read_csv(opener(linkstats_path,'rt'),  
                        sep=r'\s+', engine='python')  
    # Find all HRS...avg columns
    hrs_cols = [c for c in df_ls.columns if c.startswith('HRS') and c.endswith('avg')]  
    # Set index by LINK.  
    df_ls = df_ls.set_index('LINK')  
    labels_np = np.zeros((num_links, len(hrs_cols)), dtype=np.float32)  
    for lid, idx in id2idx.items():  
        key = int(lid)  
        if key in df_ls.index:  
            labels_np[idx] = df_ls.loc[key, hrs_cols].values  
        # else: Keep at 0 

    labels = tf.constant(labels_np, dtype=tf.float32)  

    # ————————————————  
    # 3) (Optional) Read baselinkstats and add the extra feature base_hrs_avg  
    base_feat_np = None  
    if base_linkstats_path:  
        opener = gzip.open if base_linkstats_path.endswith('.gz') else open  
        df_b = pd.read_csv(opener(base_linkstats_path,'rt'),  
                           sep=r'\s+', engine='python')  
        hrs_b = [c for c in df_b.columns if c.startswith('HRS') and c.endswith('avg')]  
        df_b = df_b.set_index('LINK')  
        base_feat_np = np.zeros((num_links, len(hrs_b)), dtype=np.float32)  
        for lid, idx in id2idx.items():  
            key = int(lid)  
            if key in df_b.index:  
                base_feat_np[idx] = df_b.loc[key, hrs_b].values  
        # Convert to tf.Tensor
        base_feat = tf.constant(base_feat_np, dtype=tf.float32)  
    else:  
        # Fill in zero if not provided.  
        base_feat = tf.zeros((num_links, len(hrs_cols)), dtype=tf.float32)  

    # ————————————————  
    # 4) Constructing the line graph edges: link_i.to == link_j.from → i→j 
    from_map = {}  
    for i,lk in enumerate(link_list):  
        from_map.setdefault(lk['from'], []).append(i)  

    src, dst = [], []  
    for i,lk in enumerate(link_list):  
        for j in from_map.get(lk['to'], []):  
            src.append(i); dst.append(j)  

    src = tf.constant(src, dtype=tf.int32)  
    dst = tf.constant(dst, dtype=tf.int32)  

    # ————————————————  
    # 5) Node features 
    node_features = {  
        'length':    tf.constant([[lk['length']]    for lk in link_list], tf.float32),  
        'freespeed': tf.constant([[lk['freespeed']] for lk in link_list], tf.float32),  
        'capacity':  tf.constant([[lk['capacity']]  for lk in link_list], tf.float32),  
        'permlanes': tf.constant([[lk['permlanes']] for lk in link_list], tf.float32),  
        'xf':        tf.constant([[lk['xf']]        for lk in link_list], tf.float32),  
        'yf':        tf.constant([[lk['yf']]        for lk in link_list], tf.float32),  
        'xt':        tf.constant([[lk['xt']]        for lk in link_list], tf.float32),  
        'yt':        tf.constant([[lk['yt']]        for lk in link_list], tf.float32),  
        # Treat the 25 `avg` values from `baselinkstats` as a vector feature.  
        'base_hrs_avg': base_feat,   # shape [num_links,25]  
    }  

    adjacency = tfgnn.Adjacency.from_indices(  
                    source=('links', src),  
                    target=('links', dst)  
                )

    #print("adj = ",adjacency.source)
    #print(adjacency.source.shape[0])

    edge_features = {
        "edge_feat": tf.zeros([adjacency.source.shape[0], 1], dtype=tf.float32)
    }


    # ————————————————  
    # 6) Construct a GraphTensor and return it.  
    graph = tfgnn.GraphTensor.from_pieces(  
        node_sets={  
            'links': tfgnn.NodeSet.from_fields(  
                sizes=[num_links],  
                features=node_features  
            )  
        },  
        edge_sets={  
            'line_graph': tfgnn.EdgeSet.from_fields(  
                sizes=[src.shape[0]],  
                adjacency=adjacency,
                features=edge_features  
            )  
        }  
    )   

    return graph, labels  

# ==== Usage Examples ====  
# gt, labels = build_line_graph_tensor("network.xml.gz", "linkstats.txt.gz","base_linkstats.txt.gz")  
# print(gt)  
# print(labels.shape)    # (num_links, 25)  




def prepare_training_data(  
    dir_path: str,  
    network_pattern_mid="network_output",  
    network_pattern_end=".xml.gz",  
    ls_pattern_mid=".linkstats_output",  
    ls_pattern_end=".txt.gz"  
) -> List[Tuple[tfgnn.GraphTensor, tf.Tensor]]:  
    # 1) glob linkstats 
    pat_ls = os.path.join(dir_path, f"*{ls_pattern_mid}*{ls_pattern_end}")  
    ls_files = glob.glob(pat_ls)  
    if not ls_files:  
        raise FileNotFoundError(f"No linkstats files under {pat_ls}")  

    # 1.1) Take out base.
    base_glob = os.path.join(dir_path, f"*{ls_pattern_mid}_base{ls_pattern_end}")  
    base_ls_list = glob.glob(base_glob)  
    assert len(base_ls_list)==1, f"Expect exactly one base linkstats, got {base_ls_list}"  
    base_ls = base_ls_list[0]  

    # 1.2) Map the remaining files as M -> path.
    rex_ls = re.compile(re.escape(ls_pattern_mid) + r"(.+?)" + re.escape(ls_pattern_end) + r"$")  
    other_ls_map = {}  
    for fn in ls_files:  
        if fn == base_ls:  
            continue  
        name = os.path.basename(fn)  
        m = rex_ls.search(name)  
        if m:  
            other_ls_map[m.group(1)] = fn  
    if not other_ls_map:  
        raise ValueError("No non-base linkstats files found")  

    # 2) glob network file
    pat_net = os.path.join(dir_path, f"{network_pattern_mid}*{network_pattern_end}")  
    net_files = glob.glob(pat_net)  

    # 2.1) Base network (optional)  
    base_net_list = glob.glob(  
        os.path.join(dir_path, f"{network_pattern_mid}_base{network_pattern_end}")  
    )  
    base_net = base_net_list[0] if len(base_net_list)==1 else None  

    # 2.2) The remaining network maps M to paths.
    rex_net = re.compile(re.escape(network_pattern_mid) + r"(.+?)" + re.escape(network_pattern_end) + r"$")  
    other_net_map = {}  
    for fn in net_files:  
        if fn == base_net:  
            continue  
        name = os.path.basename(fn)  
        m = rex_net.search(name)  
        if m:  
            other_net_map[m.group(1)] = fn  

    
    results = []  
    for m, ls_path in sorted(other_ls_map.items(), key=lambda x: x[0]):  
        net_path = other_net_map.get(m)  
        assert net_path, f"Missing network file for M={m}"  
        graph, labels = build_line_graph_tensor(  
            network_path=net_path,  
            linkstats_path=ls_path,  
            base_linkstats_path=base_ls  
        )  
        results.append((graph, labels))  

    

    return results 



def build_dataset(data_tensors:List[Tuple[tfgnn.GraphTensor, tf.Tensor]])->tf.data.Dataset:
    # Further wrap it into a tf.data.Dataset: 
    example_graph, example_labels = data_tensors[0]  
    graph_spec  = example_graph.spec
    labels_spec = tf.TensorSpec(  
        shape=example_labels.shape,  
        dtype=example_labels.dtype,  
        name="labels"  
    )  
    ds = tf.data.Dataset.from_generator(  
        lambda: ((g, l) for g,l in data_tensors),  
        output_signature=(  
          graph_spec,  
        labels_spec 
        )  
    )

    return ds


def min_max_nor_lab(lt,lmin,lmax):
    #minMax normalization for label
    if lmin==lmax:
        return tf.ones_like(lt)
    else:
        return (lt-lmin)/(lmax-lmin)

def min_max_nor_feat(gt,fmin,fmax,feat_name):
    #minMax normalization for feature
    if fmin==fmax:
        ft_nor = tf.ones_like(gt.node_sets['links'][feat_name]) 
    else:
        ft_nor = (gt.node_sets['links'][feat_name] -fmin) / (fmax - fmin)

    
    feat_dict = dict(gt.node_sets['links'].features)
    feat_dict[feat_name] = ft_nor
    return gt.replace_features(  
    node_sets={  
      "links": feat_dict 
    }  
  )  

def min_max_nor_sample(st, info_dict):  
    """  
    Perform min-max normalization on a single sample (gt, lt).

    st: tuple, (gt, lt)
    - gt: GraphTensor, containing several features of node_sets['links']
    - lt: Tensor, label
    info_dict: dict, mapping each feature name or 'label' to {'min': ..., 'max': ...}

    Returns:
    gt_nor, lt_nor 
    """  
    gt, lt = st  
  

    # Normalize in sequence.
    for key, stats in info_dict.items():  
        vmin = stats['min']  
        vmax = stats['max']  
        
        # Normalized label 
        if key == 'label':  
            lt = min_max_nor_lab(lt, vmin, vmax)  
        # Normalize a feature of `node_sets['links']` in the gt.  
        else:  
            
            gt = min_max_nor_feat(gt, vmin, vmax, feat_name=key)  
    
    return gt, lt  

def get_data_info(data):  
    """  
    data: list of (gt, lt) tuples
  - gt: GraphTensor, node_sets['links'].features contains several [N_i, 1] Tensors
  - lt: Tensor or a Tensor of shape [1,]

    Returns:
  info: dict, of the form
    {
      'feat1': {'min':…, 'max':…, 'avg':…, 'var':…, 'std':…},
      'feat2': {...},
      'label': {...}
    }
    """  
    # Get all feature names. 
    first_gt, _ = data[0]  
    feat_names = list(first_gt.node_sets['links'].features.keys())  
    
    # Initialize the collection container  
    feat_vals = {name: [] for name in feat_names}  
    label_vals = []  
    
    # Initialize the collection container 
    for gt, lt in data:  
        # Collect labels  
        lt_arr = lt.numpy().reshape(-1)        # Scalar and shape=(1,) can both be flattened.
        label_vals.extend(lt_arr.tolist())  
        
        # Collect all features.  
        node_feats = gt.node_sets['links'].features  
        for name in feat_names:  
            v = node_feats[name].numpy().reshape(-1)  
            feat_vals[name].extend(v.tolist())  
    
    # Calculate statistics 
    info = {}  
    # —— label  
    arr = np.array(label_vals, dtype=np.float64)  
    info['label'] = {  
        'min': float(arr.min()),  
        'max': float(arr.max()),  
        'avg': float(arr.mean()),  
        'var': float(arr.var()),  
        'std': float(arr.std()),  
    }  
    # —— all features  
    for name, lst in feat_vals.items():  
        arr = np.array(lst, dtype=np.float64)  
        info[name] = {  
            'min': float(arr.min()),  
            'max': float(arr.max()),  
            'avg': float(arr.mean()),  
            'var': float(arr.var()),  
            'std': float(arr.std()),  
        }  
    
    return info

def min_max_nor_data(data,info):
    '''
    data: list of (gt, lt) tuples
    - gt: GraphTensor, node_sets['links'].features contains several [N_i, 1] Tensors
    - lt: Tensor or a Tensor of shape [1,]

    Returns:
    the list of (gt_nor, lt_nor) tuples, i.e. the normalized data
    '''
    data_nor = list()
    for st in data:
        data_nor.append(min_max_nor_sample(st, info))
    return data_nor
# Exemple
# data = prepare_training_data("berlin_extracted_train",
# network_pattern_mid="network_berlin", ls_pattern_mid=".linkstats_berlin")  
# info = get_data_info(data)
# data_nor =  min_max_nor_data(data,info)
# data_val_nor = min_max_nor_data(data_val,info)
# ds_nor = build_dataset(data_nor)
# ds_val_nor = build_dataset(data_val_nor)
# train_dataset_nor = ds_nor
# batch_size = 32
# train_ds_batched_nor = train_dataset_nor.shuffle(buffer_size=3).batch(batch_size=batch_size).repeat()


