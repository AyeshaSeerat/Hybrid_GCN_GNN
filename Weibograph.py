# -*- coding: utf-8 -*-
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os

cwd = os.getcwd()

class NodeTweet:
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

def str2matrix(Str):
    """
    Convert a string of "index:freq" pairs to word frequency and index arrays.
    """
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq = float(pair.split(':')[1])
        index = int(pair.split(':')[0])
        if index <= 5000:
            wordFreq.append(freq)
            wordIndex.append(index - 1)  # Convert to zero-based index
    return wordFreq, wordIndex

def constructMat(tree):
    """
    Construct feature matrices and edge lists for a given tree.
    """
    index2node = {i: NodeTweet(idx=i) for i in tree}
    rootindex, root_index, root_word = None, None, None

    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq

        if indexP != 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        else:
            rootindex = indexC - 1
            root_index = wordIndex
            root_word = wordFreq

    # Root node feature
    rootfeat = np.zeros([1, 5000], dtype=np.float32)  # Use float32
    if root_index:
        rootfeat[0, np.array(root_index)] = np.array(root_word, dtype=np.float32)

    # Generate edge lists
    row, col = [], []
    x_word, x_index = [], []
    for node_id, node in index2node.items():
        for child in node.children:
            row.append(node_id - 1)
            col.append(child.idx - 1)
        x_word.append(node.word)
        x_index.append(node.index)

    return x_word, x_index, (row, col), rootfeat, rootindex

def getfeature(x_word, x_index):
    """
    Generate feature matrix for nodes.
    """
    x = np.zeros([len(x_index), 5000], dtype=np.float32)  # Use float32
    for i, indices in enumerate(x_index):
        if indices:
            x[i, np.array(indices)] = np.array(x_word[i], dtype=np.float32)
    return x

def main(obj="Weibo"):
    treePath = os.path.join(cwd, 'data/Weibo/weibotree.txt')
    print("Reading Weibo tree...")
    treeDic = {}
    with open(treePath) as f:
        for line in f:
            line = line.rstrip()
            eid, indexP, indexC, Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]), line.split('\t')[3]
            if eid not in treeDic:
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
    print('Total trees:', len(treeDic))

    labelPath = os.path.join(cwd, "data/Weibo/weibo_id_label.txt")
    print("Loading Weibo labels...")
    labelDic = {}
    event = []
    with open(labelPath) as f:
        for line in f:
            line = line.rstrip()
            eid, label = line.split(' ')[0], line.split(' ')[1]
            labelDic[eid] = int(label)
            event.append(eid)

    print(f"Total labels: {len(labelDic)}, Total events: {len(event)}")

    # Create output directory
    weibograph_dir = os.path.join(cwd, 'data/Weibograph')
    os.makedirs(weibograph_dir, exist_ok=True)

    def loadEid(tree, eid, label):
        """
        Process a single tree and save its data to file.
        """
        if tree is None:
            return None
        try:
            x_word, x_index, edge_index, rootfeat, rootindex = constructMat(tree)
            x_x = getfeature(x_word, x_index)
            np.savez(
                os.path.join(weibograph_dir, eid + '.npz'),
                x=x_x,
                root=rootfeat,
                edgeindex=np.array(edge_index, dtype=np.int64),  # Use int64 for edge indices
                rootindex=rootindex,
                y=np.array([label], dtype=np.int64)  # Use int64 for labels
            )
        except MemoryError:
            print(f"MemoryError: Skipping tree {eid} due to large size.")
        except Exception as e:
            print(f"Error processing tree {eid}: {e}")

    print("Processing dataset...")
    Parallel(n_jobs=4, backend='threading')(  # Reduce n_jobs for lower memory usage
        delayed(loadEid)(treeDic.get(eid), eid, labelDic[eid]) for eid in tqdm(event)
    )

if __name__ == '__main__':
    main()
cd 