# -*- coding: utf-8 -*-
import sys
import random
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import networkx as nx
import math
import operator
sys.path.append(os.getcwd()) #add the env path
from main5 import train

from config5 import DRUG_EXAMPLE, RESULT_LOG, PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR, ENTITY2ID_FILE, KG_FILE, \
    EXAMPLE_FILE,  DRUG_VOCAB_TEMPLATE, ENTITY_VOCAB_TEMPLATE, PATH_VOCAB_TEMPLATE, \
    RELATION_VOCAB_TEMPLATE, SEPARATOR, TRAIN_DATA_TEMPLATE, DEV_DATA_TEMPLATE, ADJ_PATH_TEMPLATE, \
    TEST_DATA_TEMPLATE, ADJ_ENTITY_TEMPLATE, ADJ_RELATION_TEMPLATE, ADJ_ENTITY_ALL_INFO_TEMPLATE, ModelConfig, NEIGHBOR_SIZE
from utils import pickle_dump, format_filename,write_log,pickle_load

'''
net: {'a':{'b':[c, d], 'e':[f]}, 'a1':{'b1':[c1, d1], 'e1':[f1]}, ...}
'''
def addTriple(net, source, target, edge):
    if source in net:
        if target in net[source]:
            net[source][target].add(edge)
        else:
            net[source][target] = set([edge])
    else:
        net[source] = {}
        net[source][target] = set([edge])

    return net

def getLinks(net, source):
    if source not in net:
        return {}
    return net[source]

def BiRandomNWalk(triples, center_node, neighbor_size, path_num, path_depth, add_relation, pr):

    threshold = 10   # math.ceil()
    ar = 2 / 3    # 1 / 2     2 / 3    3 / 4  OGB（12 1/2） KEGG(10 2/3)  DrugBank(12 1/2)
    
    paths = []
    center_node_neighs_entity = []
    center_node_neighs_relation = []
    for _ in range(path_depth):
        center_node_neighs_entity.append([])
    for _ in range(path_depth):
        center_node_neighs_relation.append([])

    path = 'n' + str(center_node)
    next_node = center_node
    '''
        net: {'a': {'b': [c, d], 'e': [f]}, 'a1': {'b1': [c1, d1], 'e1': [f1]}, ...}
        return net[source]
    '''
    neighs = getLinks(triples, str(next_node))
    neighs_num = len(neighs)
    if neighs_num == 0:
        neighs = {center_node: [add_relation]}
        neighs_num = 1

    neighs_1hop = []
    pr_values = {}
    if neighs_num < neighbor_size:
        for neigh in neighs:
            pr_values[neigh] = pr[int(neigh)]

        if neighs_num > threshold:
            num_important_node = math.ceil(threshold * ar)
            neighs_1hop.extend([key[0] for key in sorted(pr_values.items(), key=lambda x: x[1], reverse=True)[: num_important_node]])
            neighs_1hop.extend(np.random.choice(neighs_1hop, neighbor_size-num_important_node, replace=True))
        else:
            neighs_1hop.extend([key[0] for key in sorted(pr_values.items(), key=lambda x: x[1], reverse=True)[: neighs_num]])
            neighs_1hop.extend(np.random.choice(neighs_1hop, neighbor_size-neighs_num, replace=True))

    else:
        for neigh in neighs:
            pr_values[neigh] = pr[int(neigh)]
        if neighs_num > threshold:
            num_important_node = math.ceil(threshold * ar)
            if neighbor_size > num_important_node:
                neighs_1hop.extend([key[0] for key in sorted(pr_values.items(), key=lambda x: x[1], reverse=True)[: num_important_node]])
                neighs_1hop.extend(np.random.choice(neighs_1hop, neighbor_size - num_important_node, replace=True))
            else:
                neighs_1hop.extend([key[0] for key in sorted(pr_values.items(), key=lambda x: x[1], reverse=True)[: neighbor_size]])

        else:
            neighs_1hop.extend([key[0] for key in sorted(pr_values.items(), key=lambda x: x[1], reverse=True)[: neighbor_size]])

    neighs_sub_1hop = dict()
    for i in range(path_num):
        path_end = ''
        neighs_sub_1hop[i] = random.sample(neighs_1hop, 2)
        edges1 = []
        edges2 = []
        for edge1 in neighs[neighs_sub_1hop[i][0]]:
            edges1.append(edge1)
        edge1_temp = str(random.choice(edges1))
        for edge2 in neighs[neighs_sub_1hop[i][1]]:
            edges2.append(edge2)
        edge2_temp = str(random.choice(edges2))
        path_follow = 'n' + str(neighs_sub_1hop[i][0]) + '<-' + 'e' + edge1_temp + '<-' + path \
               + '->' + 'e' + edge2_temp + '->' + 'n' + str(neighs_sub_1hop[i][1])
        center_node_neighs_entity[0].extend([int(neighs_sub_1hop[i][0]), int(neighs_sub_1hop[i][1])])
        center_node_neighs_relation[0].extend([int(edge1_temp), int(edge2_temp)])
        for j in neighs_sub_1hop[i]:
            neighs_1hop.remove(j)

        next_node1 = neighs_sub_1hop[i][0]
        next_node2 = neighs_sub_1hop[i][1]
        for j in range(path_depth-1):
            neighs_sub1 = getLinks(triples, next_node1)
            queue1 = []
            pr1_values = {}
            if len(neighs_sub1) == 0:
                neighs_sub1[next_node1] = {add_relation}
                for neigh in neighs_sub1:
                    pr1_values[neigh] = pr[int(neigh)]
            else:
                for neigh in neighs_sub1:
                    pr1_values[neigh] = pr[int(neigh)]
            neigh1 = max(pr1_values.items(), key=operator.itemgetter(1))[0]
            for edge in neighs_sub1[neigh1]:
                queue1.append((edge, neigh1))
            edge1, next_node1 = random.choice(queue1)
            neighs_sub2 = getLinks(triples, next_node2)
            queue2 = []
            pr2_values = {}
            if len(neighs_sub2) == 0:
                neighs_sub2[next_node2] = {add_relation}
                for neigh in neighs_sub2:
                    pr2_values[neigh] = pr[int(neigh)]
            else:
                for neigh in neighs_sub2:
                    pr2_values[neigh] = pr[int(neigh)]
            neigh2 = max(pr2_values.items(), key=operator.itemgetter(1))[0]
            for edge in neighs_sub2[neigh2]:
                queue2.append((edge, neigh2))
            edge2, next_node2 = random.choice(queue2)

            path_end = 'n' + str(next_node1) + '<-' + 'e' + str(edge1) + '<-' + path_follow \
                   + '->' + 'e' + str(edge2) + '->' + 'n' + str(next_node2)
            center_node_neighs_entity[j+1].extend([int(next_node1), int(next_node2)])
            center_node_neighs_relation[j+1].extend([int(edge1), int(edge2)])
        paths.append(path_end)
    paths = list(paths)

    center_node_neighs_entity_end = []
    for i in range(path_depth):
        # center_node_neighs_entity_end : [ 4(1), 4(2) ]
        center_node_neighs_entity_end.extend(center_node_neighs_entity[i])

    center_node_neighs_relation_end = []
    for i in range(path_depth):
        # center_node_neighs_entity_end : [ 4(1), 4(2) ]
        center_node_neighs_relation_end.extend(center_node_neighs_relation[i])

    # return paths, center_node_neighs_info_end
    return paths, center_node_neighs_entity_end, center_node_neighs_relation_end


def saveWalksToFile(filename, all_lines):
    outF = open(filename, "w")
    outF.writelines(all_lines)
    outF.close()


def read_entity2id_file(file_path: str, drug_vocab: dict, entity_vocab: dict, drug_entity: dict, path_vocab: dict, dataset: str):
    print(f'Logging Info - Reading entity2id file: {file_path}' )
    assert len(drug_vocab) == 0 and len(entity_vocab) == 0 and len(path_vocab) == 0
    with open(file_path, encoding='utf8') as reader:
        count = 0
        path_index = 0
        for line in reader:
            if count == 0:
                count += 1
                continue
            drug, entity = line.strip().split('\t')
            drug_vocab[entity] = len(drug_vocab)
            entity_vocab[entity] = len(entity_vocab)
            if dataset is 'kegg':
                if drug.startswith('<http://bio2rdf.org/kegg:D') and drug[-1] == '>' and len(drug) == 32:
                    drug_entity[entity] = len(entity_vocab)-1
                    path_vocab[str(path_index)] = len(path_vocab)
                    path_index += 1
            elif dataset is 'drugbank':
                if drug.startswith('<http://bio2rdf.org/drugbank:DB') and drug[-1] == '>' and len(drug) == 37:
                    drug_entity[entity] = len(entity_vocab)-1
                    path_vocab[str(path_index)] = len(path_vocab)
                    path_index += 1
            elif dataset is 'ogb':
                if drug.startswith('CID'):
                    drug_entity[entity] = len(entity_vocab)-1
                    path_vocab[str(path_index)] = len(path_vocab)
                    path_index += 1


def read_example_file(file_path: str, separator: str, drug_vocab: dict):
    print(f'Logging Info - Reading example file: {file_path}')
    assert len(drug_vocab)>0
    examples = []
    with open(file_path,encoding='utf8') as reader:
        for idx, line in enumerate(reader):
            d1, d2, flag = line.strip().split(separator)[:3]
            if d1 not in drug_vocab or d2 not in drug_vocab:
                continue
            if d1 in drug_vocab and d2 in drug_vocab:
                examples.append([drug_vocab[d1],drug_vocab[d2],int(flag)])
    examples_matrix = np.array(examples)
    print(f'size of example: {examples_matrix.shape}')
    return examples_matrix

def read_kg(file_path: str, entity_vocab: dict, relation_vocab: dict, drug_entity: dict, neighbor_sample_size: int, path_num: int, n_depth: int, triples: dict):
    print(f'Logging Info - Reading kg file: {file_path}')

    print('len_entity_vocab:', len(entity_vocab))    # 129910
    print('len_relation_vocab:', len(relation_vocab))   # 0
    with open(file_path, encoding='utf8') as reader:
        count = 0
        for line in reader:
            if count == 0:
                count += 1
                continue
            head, tail, relation = line.strip().split(' ')

            if head not in entity_vocab:
                entity_vocab[head] = len(entity_vocab)
            if tail not in entity_vocab:
                entity_vocab[tail] = len(entity_vocab)
            if relation not in relation_vocab:
                relation_vocab[relation] = len(relation_vocab)
            addTriple(triples, head, tail, relation)

    relation_vocab[str(len(relation_vocab))] = len(relation_vocab)
    print(f'Logging Info - num of entities: {len(entity_vocab)}, 'f'num of relations: {len(relation_vocab)}')

    n_entity = len(entity_vocab)
    adj_entity = np.zeros(shape=(n_entity, neighbor_sample_size*n_depth), dtype=np.int64)
    adj_relation = np.zeros(shape=(n_entity, neighbor_sample_size*n_depth), dtype=np.int64)
    adj_path = np.zeros(shape=(n_entity, 1), dtype=np.int64)

    # 实体id列表：[7, 72, 96...]
    drug_entities = list(drug_entity.values())

    print("开始运行...")
    edges = []
    with open(file_path, 'r', encoding='utf8') as f:
        count = 0
        lines = f.readlines()
        for line in lines:
            if count == 0:
                count += 1
                continue
            h, t, r = line.strip("\n").split(" ")
            edges.append((int(h), int(t)))

    num_node = len(entity_vocab)
    G = nx.DiGraph()
    G.add_nodes_from(range(0, num_node))
    G.add_edges_from(edges)
    pr = nx.pagerank(G)

    filename = 'BiRandomWork-path%d-depth%d.txt' % (path_num, n_depth)
    print(filename)
    sequences = []

    i = 0
    for center_node in drug_entities:
        center_node_neighs_path = []
        # center_node_neighs_entity: [ 4(1), 4(2) ] center_node_neighs_relation: [ 4'(1), 4'(2) ]
        paths, center_node_neighs_entity, center_node_neighs_relation = BiRandomNWalk(triples, center_node, neighbor_sample_size, path_num, n_depth, len(relation_vocab)-1, pr)
        center_node_neighs_path.append(i)
        # sequences.extend([s + '\n' for s in paths])
        adj_entity[center_node] = np.array(center_node_neighs_entity)
        adj_relation[center_node] = np.array(center_node_neighs_relation)
        adj_path[center_node] = np.array(center_node_neighs_path)
        i += 1
    # saveWalksToFile(filename, sequences)

    return adj_entity, adj_relation, adj_path


def process_data(dataset: str, neighbor_sample_size: int, path_num: int, n_depth: int, embed_dim: int, l2_weight: float, lr:float, K: int):
    drug_vocab = {}
    entity_vocab = {}
    relation_vocab = {}
    drug_entity = {}
    path_vocab = {}
    triples = {}

    read_entity2id_file(ENTITY2ID_FILE[dataset], drug_vocab, entity_vocab, drug_entity, path_vocab, dataset)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, DRUG_VOCAB_TEMPLATE, dataset=dataset), drug_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset), entity_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, PATH_VOCAB_TEMPLATE, dataset=dataset), path_vocab)

    examples_file = format_filename(PROCESSED_DATA_DIR, DRUG_EXAMPLE, dataset=dataset)
    examples = read_example_file(EXAMPLE_FILE[dataset], SEPARATOR[dataset], drug_vocab)

    np.save(examples_file, examples)

    adj_entity, adj_relation, adj_path = read_kg(KG_FILE[dataset], entity_vocab, relation_vocab, drug_entity,
                                  neighbor_sample_size, path_num, n_depth, triples)

    pickle_dump(format_filename(PROCESSED_DATA_DIR, DRUG_VOCAB_TEMPLATE, dataset=dataset), drug_vocab)

    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset), entity_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, RELATION_VOCAB_TEMPLATE, dataset=dataset), relation_vocab)

    adj_entity_file = format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE, dataset=dataset)
    np.save(adj_entity_file, adj_entity)
    print('Logging Info - Saved:', adj_entity_file)

    adj_relation_file = format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE, dataset=dataset)
    np.save(adj_relation_file, adj_relation)
    print('Logging Info - Saved:', adj_relation_file)

    adj_path_file = format_filename(PROCESSED_DATA_DIR, ADJ_PATH_TEMPLATE, dataset=dataset)
    np.save(adj_path_file, adj_path)
    print('Logging Info - Saved:', adj_path_file)

    cross_validation(K, examples, dataset, neighbor_sample_size, n_depth, embed_dim, l2_weight, lr)


def cross_validation(K_fold, examples, dataset, neighbor_sample_size, n_depth, embed_dim, l2_weight, lr):
    random.seed(2284)
    subsets = dict()
    n_subsets = int(len(examples)/K_fold)
    remain = set(range(0, len(examples)-1))
    for i in reversed(range(0, K_fold-1)):

        subsets[i] = random.sample(remain, n_subsets)

        remain = remain.difference(subsets[i])

    subsets[K_fold-1] = remain

    # aggregator_types = ['sum','concat','neigh']
    aggregator_types = ['sum']
    for t in aggregator_types:
        count = 1
        temp = {'dataset': dataset, 'aggregator_type': t, 'avg_auc': 0.0, 'avg_acc': 0.0, 'avg_f1': 0.0, 'avg_aupr': 0.0}
        # fpr={}
        # tpr={}
        # auc={}
        # pre={}
        # rec={}
        # aupr={}
        for i in reversed(range(0, K_fold)):
            test_d = examples[list(subsets[i])]
            n_test_val = int(len(test_d) / 2)
            val_d = random.sample(subsets[i], n_test_val)
            test_data = set(subsets[i]).difference(val_d)
            val_d = examples[list(val_d)]
            test_data = examples[list(test_data)]

            # val_d, test_data = train_test_split(test_d, test_size=0.5)
            train_d = []
            '''
            训练集则于是取剩余则数的全部数据作为训练数据
            '''
            for j in range(0, K_fold):
                if i != j:
                    '''
                    #extend和append类似，在原列表中添加其它列表元素
                    '''
                    train_d.extend(examples[list(subsets[j])])

            train_data = np.array(train_d)
            train_log = train(
                kfold=count,
                dataset=dataset,
                train_d=train_data,
                dev_d=val_d,
                test_d=test_data,
                neighbor_sample_size=neighbor_sample_size,
                embed_dim=embed_dim,
                n_depth=n_depth,
                l2_weight=l2_weight,
                lr=lr,
                optimizer_type='adam',
                batch_size=4096,
                aggregator_type=t,
                n_epoch=50,
                callbacks_to_add=['modelcheckpoint', 'earlystopping']
            )     
            count += 1
            # break

            temp['avg_auc'] = temp['avg_auc']+train_log['test_auc']
            temp['avg_acc'] = temp['avg_acc']+train_log['test_acc']
            temp['avg_f1'] = temp['avg_f1']+train_log['test_f1']
            temp['avg_aupr'] = temp['avg_aupr']+train_log['test_aupr']
            # fpr[K_fold-i] = train_log['test_fpr']
            # tpr[K_fold-i] = train_log['test_tpr']
            # auc[K_fold-i] = train_log['test_auc']
            # pre[K_fold-i] = train_log['test_precision']
            # rec[K_fold-i] = train_log['test_recall']
            # aupr[K_fold-i] = train_log['test_aupr']
            # train_log['test_fpr'] = {}
            # train_log['test_tpr'] = {}
            # train_log['test_precision'] = {}
            # train_log['test_recall'] = {}
        for key in temp:
            if key == 'aggregator_type' or key == 'dataset':
                continue
            temp[key] = temp[key]/K_fold
        write_log(format_filename(LOG_DIR, RESULT_LOG[dataset]), temp, 'a')
        print(f'Logging Info - {K_fold} fold result: avg_auc: {temp["avg_auc"]}, avg_acc: {temp["avg_acc"]}, avg_f1: {temp["avg_f1"]}, avg_aupr: {temp["avg_aupr"]}')


if __name__ == '__main__':
    begin_time = time.time()
    print('begin_time：', time.strftime("%H:%M:%S", time.localtime(begin_time)))
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)
    # folder = 'deal_kg/kegg-drug/paths/'
    # if not os.path.isdir(folder):
    #     os.mkdir(folder)

    model_config = ModelConfig()
    process_data('kegg', NEIGHBOR_SIZE['kegg'], NEIGHBOR_SIZE['kegg']//2, model_config.n_depth, model_config.embed_dim, model_config.l2_weight, model_config.lr, 5)
    #process_data('drugbank', NEIGHBOR_SIZE['drugbank'], NEIGHBOR_SIZE['drugbank']//2, model_config.n_depth, model_config.embed_dim, model_config.l2_weight, model_config.lr, 5)
    #process_data('ogb', NEIGHBOR_SIZE['ogb'], NEIGHBOR_SIZE['ogb']//2, model_config.n_depth, model_config.embed_dim, model_config.l2_weight, model_config.lr, 5)

    finish_time = time.time()
    print('finish_time：', time.strftime("%H:%M:%S", time.localtime(finish_time)))
    elapsed_time = finish_time - begin_time
    print('elapsed_time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

