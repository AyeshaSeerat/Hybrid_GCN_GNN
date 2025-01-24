# -*- coding: utf-8 -*-
import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
cwd = os.getcwd()

class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq = float(pair.split(':')[1])
        index = int(pair.split(':')[0])
        if index <= 5000:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex

def constructMat(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            rootindex = indexC - 1
            root_index = nodeC.index
            root_word = nodeC.word
    rootfeat = np.zeros([1, 5000])
    if len(root_index) > 0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    row, col = [], []
    x_word = []
    x_index = []
    for node_id, node in index2node.items():
        for child in node.children:
            row.append(node.idx - 1)  # Convert to zero-based index
            col.append(child.idx - 1)  # Convert to zero-based index
        x_word.append(node.word)
        x_index.append(node.index)
    edgematrix = [row, col]
    return x_word, x_index, edgematrix, rootfeat, rootindex

def getfeature(x_word, x_index):
    x = np.zeros([len(x_index), 5000])
    for i in range(len(x_index)):
        if len(x_index[i]) > 0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x

def main(obj="Twitter16"):
    treePath = os.path.join(cwd, 'data/' + obj + '/data.TD_RvNN.vol_5000.txt')
    print("Reading Twitter tree...")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]

        if eid not in treeDic:
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
    print('Total events in treeDic:', len(treeDic))

    labelPath = os.path.join(cwd, "data/Twitter16/" + obj + "_label_All.txt")
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']

    print("Loading tree labels...")
    event, y = [], []
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        label = label.lower()
        event.append(eid)
        if label in labelset_nonR:
            labelDic[eid] = 0
        elif label in labelset_f:
            labelDic[eid] = 1
        elif label in labelset_t:
            labelDic[eid] = 2
        elif label in labelset_u:
            labelDic[eid] = 3
    print('Total events in label file:', len(labelDic))

    def loadEid(event, id, y):
        if id not in treeDic:
            print(f"Event {id} missing in treeDic.")
            return None
        x_word, x_index, tree, rootfeat, rootindex = constructMat(treeDic[id])
        x_x = getfeature(x_word, x_index)

        # Debugging feature and edge index output
        print(f"Event {id}: Feature Shape: {x_x.shape}, Edge Index Shape: {len(tree[0])}")

        # Create the directory if it doesn't exist
        save_dir = os.path.join(cwd, 'gen', obj + 'graph15')
        os.makedirs(save_dir, exist_ok=True)

        rootfeat, tree, x_x, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(rootindex), np.array(y)
        np.savez(os.path.join(save_dir, id + '.npz'), x=x_x, root=rootfeat, edgeindex=tree, rootindex=rootindex, y=y)

    print("Processing dataset...")
    Parallel(n_jobs=30, backend='threading')(delayed(loadEid)(treeDic.get(eid), eid, labelDic[eid]) for eid in tqdm(event))
    return

if __name__ == '__main__':
    main()
