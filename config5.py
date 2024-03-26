# -*- coding: utf-8 -*-

import os

RAW_DATA_DIR = os.getcwd()+'/raw_data'
PROCESSED_DATA_DIR = os.getcwd()+'/data'
LOG_DIR = os.getcwd()+'/log'
MODEL_SAVED_DIR = os.getcwd()+'/ckpt'

KG_FILE = {'drugbank':os.path.join(RAW_DATA_DIR,'drugbank','train2id.txt'),
           'kegg':os.path.join(RAW_DATA_DIR,'kegg','train2id.txt'),
           'ogb':os.path.join(RAW_DATA_DIR,'ogb','train2id.txt')}
ENTITY2ID_FILE = {'drugbank':os.path.join(RAW_DATA_DIR,'drugbank','entity2id.txt'),
                  'kegg':os.path.join(RAW_DATA_DIR,'kegg','entity2id.txt'),
                  'ogb':os.path.join(RAW_DATA_DIR,'ogb','entity2id.txt')}
EXAMPLE_FILE = {'drugbank':os.path.join(RAW_DATA_DIR,'drugbank','approved_example.txt'),
                'kegg':os.path.join(RAW_DATA_DIR,'kegg','approved_example.txt'),
                'ogb':os.path.join(RAW_DATA_DIR,'ogb','approved_example.txt')}

DRUG_VOCAB_TEMPLATE = '{dataset}_drug_vocab.pkl'
ENTITY_VOCAB_TEMPLATE = '{dataset}_entity_vocab.pkl'
PATH_VOCAB_TEMPLATE = '{dataset}_path_vocab.pkl'
RELATION_VOCAB_TEMPLATE = '{dataset}_relation_vocab.pkl'
ADJ_ENTITY_TEMPLATE = '{dataset}_adj_entity.npy'
ADJ_RELATION_TEMPLATE = '{dataset}_adj_relation.npy'
ADJ_PATH_TEMPLATE = '{dataset}_adj_path.npy'

ADJ_ENTITY_ALL_INFO_TEMPLATE = '{dataset}_adj_entity_all_info.npy'

SMILE_HASH = '{dataset}_smile_hash.npy'

SMILE_FP = '{dataset}_smile_fp.npy'

INDEXES_RANK_TEMPLATE = '{dataset}_indexes_rank.npy'

TRAIN_DATA_TEMPLATE = '{dataset}_train.npy'
DEV_DATA_TEMPLATE = '{dataset}_dev.npy'
TEST_DATA_TEMPLATE = '{dataset}_test.npy'
DRUG_EXAMPLE='{dataset}_examples.npy'

TRAIN_HISTORY={'drugbank':'drugbank_train_history5.txt','kegg':'kegg_train_history5.txt','ogb':'ogb_train_history5.txt'}
RESULT_LOG={'drugbank':'drugbank_result5.txt','kegg':'kegg_result5.txt', 'ogb':'ogb_result5.txt'}
PERFORMANCE_LOG = 'kegg_kgcn_performance5.log'
# PERFORMANCE_LOG = 'drugbank_kgcn_performance5.log'
# PERFORMANCE_LOG = 'ogb_kgcn_performance5.log'

SEPARATOR = {'drugbank':' ','kegg':'\t', 'ogb':' '}
NEIGHBOR_SIZE = {'drugbank': 8,'kegg':128,'ogb':16}


class ModelConfig(object):
    def __init__(self):
        self.neighbor_sample_size = 128 # neighbor sampling size  KEGG(128)
        self.embed_dim = 128  # dimension of embedding
        self.n_depth = 2    # depth of receptive field
        self.l2_weight = 1e-7  #    1e-7    1e-7
        self.lr = 1e-2  #    1e-2    2e-3
        self.batch_size = 4096
        self.aggregator_type = 'sum'
        self.n_epoch = 50
        self.optimizer = 'adam'

        self.drug_vocab_size = None
        self.entity_vocab_size = None
        self.relation_vocab_size = None
        self.path_vocab_size = None
        self.adj_entity = None
        self.adj_relation = None
        self.adj_path = None
        self.adj_entity_all_info = None
        
        self.smile_hash = None
        self.smile_fp = None
        self.adj_index_rank = None

        self.exp_name = None
        self.model_name = None

        # checkpoint configuration
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_auc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stoping configuration
        self.early_stopping_monitor = 'val_auc'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1
        # self.dataset='drug'
        self.K_Fold=5
        self.callbacks_to_add = None

        # config for learning rating scheduler and ensembler
        self.swa_start = 3

        # hash_vec_dim
        self.smile_hash_vec_dim = 256
        self.smile_fp_vec_dim = 256
        # encoder-decoder
        self.latent_dim = 256
        
        self.timestep = 4
