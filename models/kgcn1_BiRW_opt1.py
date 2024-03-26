# -*- coding: utf-8 -*-

# V1
from keras.layers import *
from keras.layers import Lambda
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K  # use computable function
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, roc_curve
import sklearn.metrics as m
from layers import Aggregator1
from callbacks import KGCNMetric
from models.base_model5 import BaseModel
import tensorflow as tf
import math

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class KGCN(BaseModel):
    def __init__(self, config):
        super(KGCN, self).__init__(config)

    def build(self):
        input_drug_one = Input(
            # [22，33] shape是(2,):他表示他是一个一维数组，数组中有两个元素
            # https://blog.csdn.net/weixin_38859557/article/details/80778820
            shape=(1,), name='input_drug_one', dtype='int64')
        input_drug_two = Input(
            shape=(1,), name='input_drug_two', dtype='int64')

        '''
        模型训练完毕后，h5文件中的实体权重矩阵是在此处初始化以及训练的
        而后面我也只是从h5文件中获取该权重矩阵得到训练数据中实体的特征向量的
        实体权重文件维度是 （25487，32）
        '''
        entity_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                     output_dim=self.config.embed_dim,
                                     embeddings_initializer='glorot_normal',
                                     embeddings_regularizer=l2(
                                         self.config.l2_weight),
                                     name='entity_embedding')
        relation_embedding = Embedding(input_dim=self.config.relation_vocab_size,
                                       output_dim=self.config.embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='relation_embedding')
        path_embedding = Embedding(input_dim=self.config.path_vocab_size,
                                   output_dim=self.config.embed_dim,
                                   embeddings_initializer='glorot_normal',
                                   embeddings_regularizer=l2(
                                       self.config.l2_weight),
                                   name='path_embedding')
        
        ##############################
        smile_embedding = Embedding(input_dim=self.config.path_vocab_size,
                                   output_dim=self.config.embed_dim,
                                   embeddings_initializer='glorot_normal',
                                   embeddings_regularizer=l2(
                                       self.config.l2_weight),
                                   name='smile_embedding')
                              
#        drug_hash = Lambda(lambda x: self.get_hash(x), name='get_smile_hash')(input_drug_one)  # (?, 4, 64)
#        drug_one_embedding = Lambda(lambda x: self.drug_embedding(x), name='get_drug_embedding')(drug_hash)

        receptive_list_drug_one = Lambda(lambda x: self.get_receptive_field(x), name='receptive_filed_drug_one')(input_drug_one)
        # [[1], [4], [4]]
        self_neighbor_entity_list_drug_one = receptive_list_drug_one[:self.config.n_depth + 1]
        # [[4],[4]]
        neighbor_relation_list_drug_one = receptive_list_drug_one[self.config.n_depth + 1:self.config.n_depth*2 + 1]
        # [1]
        neighbor_path_list_drug_one = receptive_list_drug_one[self.config.n_depth * 2 + 1:]
        
        ##############################
        smile_drug_one = neighbor_path_list_drug_one

        self_neighbor_embed_list_drug_one = [entity_embedding(neigh_ent) for neigh_ent in self_neighbor_entity_list_drug_one]
        neighbor_relation_embed_list_drug_one = [relation_embedding(neigh_rel) for neigh_rel in neighbor_relation_list_drug_one]
        neighbor_path_embed_list_drug_one = [path_embedding(neigh_path) for neigh_path in neighbor_path_list_drug_one]
        
        ##############################
        smile_embed_list_drug_one = [smile_embedding(smile) for smile in smile_drug_one]
        
        #neighbor_path_embed_list_drug_one_temp = Lambda(lambda x: self.concat_path(x),name='concat_path_one')(neighbor_path_embed_list_drug_one[0])
        #neighbor_path_embed_list_drug_one.append(neighbor_path_embed_list_drug_one_temp)

        neighbor_embedding = Lambda(lambda x: self.get_neighbor_info(x[0], x[1], x[2]),name='neighbor_embedding_drug_one')
        n_depth = self.config.n_depth
        '''
        第一层用relu，第二层用tanh
        depth of receptive field
        '''
        next_entity_embed_list = []
        next_entity_embed_list.append(self_neighbor_embed_list_drug_one[0])
        next_entity_embed_list1 = []
        flag1 = True
        # for fb in [[0, 1], [1, 0]]:
        for fb in [range(n_depth), reversed(range(n_depth))]:
            for depth in fb:  # 0, 1; 1, 0
                if flag1:
                    aggregator = Aggregator1[self.config.aggregator_type](
                        activation='tanh' if depth == 0 else 'relu',
                        #activation='relu',
                        regularizer=l2(self.config.l2_weight),
                        name=f'f_aggregator_{depth + 1}_drug_one'
                    )
                    if depth == 0:
                        neighbor_embed = neighbor_embedding([self_neighbor_embed_list_drug_one[depth + 1], neighbor_relation_embed_list_drug_one[depth], self_neighbor_embed_list_drug_one[depth]])
                        next_entity_embed = aggregator([self_neighbor_embed_list_drug_one[depth + 1], neighbor_embed, neighbor_path_embed_list_drug_one[0]])
                        next_entity_embed_list.append(next_entity_embed)
                    else:
                        neighbor_embed = neighbor_embedding([self_neighbor_embed_list_drug_one[depth + 1], neighbor_relation_embed_list_drug_one[depth], next_entity_embed_list[-1]])
                        next_entity_embed = aggregator([self_neighbor_embed_list_drug_one[depth + 1], neighbor_embed, neighbor_path_embed_list_drug_one[0]])
                        next_entity_embed_list.append(next_entity_embed)
                else:
                    aggregator = Aggregator1[self.config.aggregator_type](
                        activation='tanh' if depth == 0 else 'relu',
                        regularizer=l2(self.config.l2_weight),
                        name=f'b_aggregator_{depth + 1}_drug_one'
                    )
                    if depth == n_depth - 1:
                        neighbor_embed = neighbor_embedding([next_entity_embed_list[depth], neighbor_relation_embed_list_drug_one[depth], next_entity_embed_list[depth + 1]])
                        next_entity_embed = aggregator([next_entity_embed_list[depth], neighbor_embed, neighbor_path_embed_list_drug_one[0]])
                        next_entity_embed_list1.append(next_entity_embed)
                    else:
                        neighbor_embed = neighbor_embedding([next_entity_embed_list[depth], neighbor_relation_embed_list_drug_one[depth], next_entity_embed_list1[-1]])
                        next_entity_embed = aggregator([next_entity_embed_list[depth], neighbor_embed, neighbor_path_embed_list_drug_one[0]])
                        next_entity_embed_list1.append(next_entity_embed)
            flag1 = False

        # get receptive field
        receptive_list_drug_two = Lambda(lambda x: self.get_receptive_field(x), name='receptive_filed_drug_two')(input_drug_two)
        # [[1], [4], [4]]
        self_neighbor_entity_list_drug_two = receptive_list_drug_two[:self.config.n_depth + 1]
        # [[4],[4]]
        neighbor_relation_list_drug_two = receptive_list_drug_two[self.config.n_depth + 1:self.config.n_depth*2 + 1]
        # [1]
        neighbor_path_list_drug_two = receptive_list_drug_two[self.config.n_depth * 2 + 1:]
        
        ##############################
        smile_drug_two = neighbor_path_list_drug_two

        self_neighbor_embed_list_drug_two = [entity_embedding(neigh_ent) for neigh_ent in self_neighbor_entity_list_drug_two]
        neighbor_relation_embed_list_drug_two = [relation_embedding(neigh_rel) for neigh_rel in neighbor_relation_list_drug_two]
        neighbor_path_embed_list_drug_two = [path_embedding(neigh_path) for neigh_path in neighbor_path_list_drug_two]
        
        ##############################
        smile_embed_list_drug_two = [smile_embedding(smile) for smile in smile_drug_two]
        

        neighbor_embedding = Lambda(lambda x: self.get_neighbor_info(x[0], x[1], x[2]), name='neighbor_embedding')
        
        next_entity2_embed_list = []
        next_entity2_embed_list.append(self_neighbor_embed_list_drug_two[0])
        next_entity2_embed_list1 = []
        flag2 = True
        # for fb in [[0, 1], [1, 0]]:
        for fb in [range(n_depth), reversed(range(n_depth))]:
            for depth in fb:  # 0, 1; 1, 0
                if flag2:
                    aggregator = Aggregator1[self.config.aggregator_type](
                        activation='tanh' if depth == 0 else 'relu',
                        #activation='relu',
                        regularizer=l2(self.config.l2_weight),
                        name=f'f_aggregator_{depth + 1}_drug_two'
                    )
                    if depth == 0:
                        neighbor_embed = neighbor_embedding([self_neighbor_embed_list_drug_two[depth + 1], neighbor_relation_embed_list_drug_two[depth], self_neighbor_embed_list_drug_two[depth]])
                        next_entity_embed = aggregator([self_neighbor_embed_list_drug_two[depth + 1], neighbor_embed, neighbor_path_embed_list_drug_two[0]])
                        next_entity2_embed_list.append(next_entity_embed)
                    else:
                        neighbor_embed = neighbor_embedding([self_neighbor_embed_list_drug_two[depth + 1], neighbor_relation_embed_list_drug_two[depth], next_entity2_embed_list[-1]])
                        next_entity_embed = aggregator([self_neighbor_embed_list_drug_two[depth + 1], neighbor_embed, neighbor_path_embed_list_drug_two[0]])
                        next_entity2_embed_list.append(next_entity_embed)
                else:
                    aggregator = Aggregator1[self.config.aggregator_type](
                        activation='tanh' if depth == 0 else 'relu',
                        regularizer=l2(self.config.l2_weight),
                        name=f'b_aggregator_{depth + 1}_drug_two'
                    )
                    if depth == n_depth - 1:
                        neighbor_embed = neighbor_embedding([next_entity2_embed_list[depth], neighbor_relation_embed_list_drug_two[depth], next_entity2_embed_list[depth + 1]])
                        next_entity_embed = aggregator([next_entity2_embed_list[depth], neighbor_embed, neighbor_path_embed_list_drug_two[0]])
                        next_entity2_embed_list1.append(next_entity_embed)
                    else:
                        neighbor_embed = neighbor_embedding([next_entity2_embed_list[depth], neighbor_relation_embed_list_drug_two[depth], next_entity2_embed_list1[-1]])
                        next_entity_embed = aggregator([next_entity2_embed_list[depth], neighbor_embed, neighbor_path_embed_list_drug_two[0]])
                        next_entity2_embed_list1.append(next_entity_embed)
            flag2 = False
        

        drug1_squeeze_embed = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))([next_entity_embed_list1[-1], smile_embed_list_drug_one[0]])
        drug2_squeeze_embed = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))([next_entity2_embed_list1[-1], smile_embed_list_drug_two[0]])

        
        drug1_squeeze_embed = Lambda(lambda x: K.squeeze(x, axis=1))(drug1_squeeze_embed)
        drug2_squeeze_embed = Lambda(lambda x: K.squeeze(x, axis=1))(drug2_squeeze_embed)
        
        print('拼接过后的向量维度是：', drug1_squeeze_embed.shape)
        

        drug_drug_score = Lambda(lambda x: K.sigmoid(K.sum(x[0] * x[1], axis=-1, keepdims=True)))([drug1_squeeze_embed, drug2_squeeze_embed])

        model = Model([input_drug_one, input_drug_two], drug_drug_score)
        
        model.compile(optimizer=self.config.optimizer, loss='binary_crossentropy', metrics=['acc'])
        print('Model had been compiled...')
        return model

    def get_hash(self, drug_index):

        drug_index_rank = K.variable(self.config.adj_index_rank, name='index_rank')

        rank = K.gather(drug_index_rank, K.cast(drug_index, dtype='int64'))

        drug_smile_hash_matrix = K.variable(self.config.smile_fp, name='smile_hash')

        drug_smile_hash_vector = K.gather(drug_smile_hash_matrix, K.cast(rank, dtype='int64'))

        drug_smile_hash_emded = K.reshape(drug_smile_hash_vector, (-1, int(self.config.timestep), int(self.config.smile_hash_vec_dim / self.config.timestep)))  # (?, 4, 64)

        return drug_smile_hash_emded


    def drug_embedding(self, input_drug):

        drug_embed = input_drug  

#        encoder = Bidirectional(LSTM(128, return_sequences=False, return_state=False), merge_mode='sum')
#        encoder_outputs = encoder(drug_embed)  
#        encoder_outputs = K.reshape(encoder_outputs, (-1, 1, 128))
#             
#        decoder = LSTM(128, return_sequences=False, return_state=False)
#        decoder_outputs = decoder(encoder_outputs)
#        decoder_outputs = K.reshape(decoder_outputs, (-1, 1, 128))
#        
#        decoder = LSTM(128, return_sequences=False, return_state=False)
#        decoder_outputs = decoder(decoder_outputs) 
#        decoder_outputs = K.reshape(decoder_outputs, (-1, 1, 128))
        
     
        decoder_outputs = Dense(self.config.embed_dim, kernel_initializer='glorot_normal', activation='softmax')(drug_embed) # 激活函数softmax最合适 , kernel_initializer='glorot_normal'

        print('smiles')
        return decoder_outputs

    def get_receptive_field(self, entity):

        neigh_ent_list = [entity]
        neigh_rel_list = []
        neigh_path_list = []

        adj_entity_matrix = K.variable(self.config.adj_entity, name='adj_entity',dtype='int64')

        adj_relation_matrix = K.variable(self.config.adj_relation, name='adj_relation',dtype='int64')
        adj_path_matrix = K.variable(self.config.adj_path, name='adj_path',dtype='int64')

        n_neighbor = self.config.neighbor_sample_size
        new_entity = K.gather(adj_entity_matrix, K.cast(neigh_ent_list[-1], dtype='int64'))

        new_relation = K.gather(adj_relation_matrix, K.cast(neigh_ent_list[-1], dtype='int64'))
        new_path = K.gather(adj_path_matrix, K.cast(neigh_ent_list[-1], dtype='int64'))

        n_depth = self.config.n_depth
        new_entity = tf.split(new_entity, axis=2, num_or_size_splits=n_depth)  # [4, 4]
        new_relation = tf.split(new_relation, axis=2, num_or_size_splits=n_depth)  # [4, 4]



        for i in range(self.config.n_depth):
            neigh_ent_list.append(K.reshape(new_entity[i], (-1, n_neighbor)))
            neigh_rel_list.append(K.reshape(new_relation[i], (-1, n_neighbor)))
        neigh_path_list.append(K.reshape(new_path, (-1, 1)))
        

        return neigh_ent_list + neigh_rel_list + neigh_path_list

    def get_neighbor_info(self, drug, rel, ent):
        """Get neighbor representation.
        *外积，也叫hamanda积即对应元素相乘
        """
        if drug.shape[1] > 1:
            drug_rel_score = drug * rel
            weighted_ent = drug_rel_score * ent

            weighted_ent = K.reshape(weighted_ent, (K.shape(weighted_ent)[0], self.config.neighbor_sample_size, -1, self.config.embed_dim))
        else:
            drug_rel_score = drug * rel

            weighted_ent = drug_rel_score * ent

            weighted_ent = K.reshape(weighted_ent, (K.shape(weighted_ent)[0], -1, self.config.neighbor_sample_size, self.config.embed_dim))

        neighbor_embed = K.sum(weighted_ent, axis=2)

        return neighbor_embed

    def add_metrics(self, x_train, y_train, x_valid, y_valid):
        self.callbacks.append(KGCNMetric(x_train, y_train, x_valid, y_valid,
                                         self.config.aggregator_type, self.config.dataset, self.config.K_Fold))

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.callbacks = []
        self.add_metrics(x_train, y_train, x_valid, y_valid)
        '''
        hdf5文件在此处生成，是keras中的机制自行生成的
        '''
        self.init_callbacks()

        print('Logging Info - Start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size,
                       epochs=self.config.n_epoch, validation_data=(x_valid, y_valid),
                       callbacks=self.callbacks)
        '''
        没训练完就报错！！！
        没训练完就报错！！！
        没训练完就报错！！！
        '''
        print('Logging Info - training end...')

    def predict(self, x):
        return self.model.predict(x).flatten()

    def score(self, x, y, threshold=0.5):
        y_true = y.flatten()
        y_pred = self.model.predict(x).flatten()
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)

        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred)

        precision, recall, t = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        aupr = m.auc(recall, precision)
        y_pred = [1 if prob >= threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)

        return auc, acc, f1, aupr, fpr, tpr, precision, recall
        
    def concat_path(self, path):
        path_tmp = path
        for _ in range(int(math.log(self.config.neighbor_sample_size, 2))):
            path_tmp = K.concatenate([path_tmp, path_tmp], axis=1)
        return path_tmp



# a = [1, 2, 3, 4]
# a[0] = 5
# print(a)
