import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
import pickle
from .resnet import *
from .deepLearningUtils import *
from .layers import *

class Model:
    def __init__(self,
                 obt_size,
                 tbt_size,
                 bert_size=768,
                 prot_size=None,
                 name=None,
                 num_chunks=8,
                 channel=256,
                 optimizer="momentum",
                 loss_weight=[1.0, 0.25, 10.0],
                 no_last_dilation=True,
                 label_smoothing=False,
                 bert=False,
                 nretype=20,
                 partial_instance_norm=True,
                 transpose_matrix=False,
                 self_attention=False,
                 verbose=False):

        # Defining network architecture
        self.obt_size = obt_size
        self.bert_size = bert_size
        self.tbt_size = tbt_size
        self.prot_size = prot_size
        self.loss_weight = loss_weight
        
        self.no_last_dilation = no_last_dilation
        self.label_smoothing = label_smoothing
        self.nretype = nretype
        self.partial_instance_norm = partial_instance_norm
        self.transpose_matrix = transpose_matrix
        self.self_attention = self_attention
        self.bert = bert
        
        # Other network definining parameters
        self.num_chunks = num_chunks
        self.channel = channel
        self.optimizer = optimizer
        self.verbose = verbose

        # Reset all existing graphs and rebuild them.
        # Allow gpu memory growth to combat error.
        tf.reset_default_graph()
        self.built = False
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sesh = tf.Session(config=config)
        self.ops = self.build()
        self.sesh.run(tf.global_variables_initializer())

        # Keep track of training parameters.
        self.e = 0
        self.b = 0
        self.name = name
        self.loss = {"train":[], "valid":[], "valid_e":[], "valid_l":[], "valid_m":[]}

    # Building a graph
    def build(self):
        if self.built: return -1
        else: self.built = True
            
        with tf.name_scope('input'):
            # 1D convolution part
            obt_in = tf.placeholder(tf.float32, shape=[self.prot_size, self.obt_size])
            nres = tf.shape(obt_in)[0]
            
            if self.bert:
                bert_in = tf.placeholder(tf.float32, shape=[self.prot_size, self.bert_size])
            
            # 2D convolution part
            tbt_in = tf.placeholder(tf.float32, shape=[self.prot_size, self.prot_size, self.tbt_size])
            tbt = tf.expand_dims(tbt_in, axis=0)
            
            # 3D convolution part
            idx = tf.placeholder(dtype=tf.int32, shape=(None, 5))
            val = tf.placeholder(dtype=tf.float32, shape=(None))
            grid3d = tf.scatter_nd(idx, val, [nres, 24, 24, 24, self.nretype])
            
            # Training parameters
            dropout_rate = tf.placeholder_with_default(0.15, shape=()) 
            isTraining = tf.placeholder_with_default(False, shape=()) 
            learning_rate = tf.placeholder_with_default(0.001, shape=())

            # Target values
            estogram_in = tf.placeholder(tf.float32, shape=[self.prot_size, self.prot_size, 15])
            estogram = tf.expand_dims(estogram_in, axis=0)
            mask_in = tf.placeholder(tf.float32, shape=[self.prot_size, self.prot_size])
            mask = tf.expand_dims(mask_in, axis=0)
        
        layers=[]
        with tf.name_scope('3d_conv'):
            # retyper: 1x1x1 convolution
            layers.append(tf.layers.conv3d(grid3d, 20, 1, padding='same', use_bias=False))

            # 1st conv3d & batch_norm & droput & activation
            layers.append(tf.layers.conv3d(layers[-1], 20, 3, padding='valid', use_bias=True))
            layers.append(tf.keras.layers.Dropout(rate=dropout_rate)(layers[-1], training=isTraining))
            layers.append(tf.nn.elu(layers[-1]))

            # 2nd conv3d & batch_norm & activation
            layers.append(tf.layers.conv3d(layers[-1], 30, 4, padding='valid', use_bias=True))
            layers.append(tf.nn.elu(layers[-1]))

            # 3rd conv3d & batch_norm & activation
            layers.append(tf.layers.conv3d(layers[-1], 10, 4, padding='valid', use_bias=True))
            layers.append(tf.nn.elu(layers[-1]))

            # average pooling
            layers.append(tf.layers.average_pooling3d(layers[-1], pool_size=4, strides=4, padding='valid'))
            
        with tf.name_scope('2d_conv'):
            
            # Concat 3D conv output with 1D and project down to 60 dims with l layer of 1D convolution.
            layers.append(tf.layers.flatten(layers[-1]))
            layers.append(tf.concat([layers[-1], obt_in], axis=1))
            layers.append(tf.expand_dims(layers[-1], 0))
            layers.append(tf.layers.conv1d(layers[-1], self.channel//2, 1, padding='SAME'))
            layers.append(tf.nn.elu(layers[-1]))
            
            if self.bert:
                temp = tf.expand_dims(bert_in, 0)
                compressed_bert = tf.nn.elu(tf.layers.conv1d(temp, self.channel//8, 1)) 
                layers.append(tf.concat([layers[-1], compressed_bert], axis=-1))
                layers.append(tf.layers.conv1d(layers[-1], self.channel//2, 1))
                layers.append(tf.nn.elu(layers[-1]))
            
            # Put them together with tbt with self.tbt_size
            tbt = tf.concat([tf.tile(layers[-1][:,:,None,:], [1,1,nres,1]),
                            tf.tile(layers[-1][:,None,:,:], [1,nres,1,1]),
                            tbt], axis=-1)
            
            # Do instance normalization after training 
            layers.append(tf.reshape(tbt, [1, nres, nres, self.tbt_size+self.channel]))
            layers.append(tf.layers.conv2d(layers[-1], self.channel, 1, padding='SAME'))
            layers.append(tf.contrib.layers.instance_norm(layers[-1]))
            layers.append(tf.nn.elu(layers[-1]))
            
            # Resnet prediction with alpha fold style
            resnet_output = build_resnet(_input  = layers[-1],
                                         channel = self.channel, 
                                         num_chunks = self.num_chunks,
                                         require_in = self.partial_instance_norm, 
                                         isTraining = isTraining,
                                         no_last_dilation = False)
            
            layers.append(tf.nn.elu(resnet_output))
            
            if self.self_attention:
                layers.append(pixelSelfAttention(layers[-1], maxpool=3))
            
            # Resnet prediction for errorgram branch
            error_predictor = build_resnet(_input  = layers[-1],
                                           channel = self.channel,
                                           num_chunks = 1,
                                           require_in = False, 
                                           transpose_matrix=self.transpose_matrix,
                                           isTraining = isTraining,
                                           no_last_dilation = self.no_last_dilation)
            
            error_predictor = tf.nn.elu(error_predictor)
            logits_error = tf.layers.conv2d(error_predictor, filters=15, kernel_size=(1,1))
            logits_error = (logits_error+tf.transpose(logits_error, [0,2,1,3]))/2
            estogram_predicted = tf.nn.softmax(logits_error)[0]
            
            # Resnet prediction for errorgram branch
            mask_predictor = build_resnet(_input = layers[-1],
                                          channel = self.channel,
                                          num_chunks = 1,
                                          require_in = False, 
                                          transpose_matrix=self.transpose_matrix,
                                          isTraining = isTraining,
                                          no_last_dilation = self.no_last_dilation)
            
            mask_predictor = tf.nn.elu(mask_predictor)
            logits_mask = tf.layers.conv2d(mask_predictor, filters=1, kernel_size=(1,1))[:, :, :, 0]
            logits_mask = (logits_mask+tf.transpose(logits_mask, [0,2,1]))/2
            mask_predicted = tf.nn.sigmoid(logits_mask)[0]
            
            # Lddt calculations
            lddt_predicted = self.calculate_LDDT(estogram_predicted, mask_predicted)
            lddt = self.calculate_LDDT(estogram_in, mask_in)
            
        # Crossentropy over all entries taken reduce mean at the end.
        with tf.name_scope("cost"):
            # Estogram evaluation via Crossentropy
            if self.verbose: print("Check 1:", estogram.shape, logits_error.shape)
            estogram_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=estogram, logits=logits_error, axis=-1)
            
            # Auxiliary estogram
            if self.verbose: print("Check 2:", mask.shape, logits_mask.shape)
            mask_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=logits_mask)

            # Lddt evaluation via MSE
            if self.verbose: print("Check 3:", lddt.shape, lddt_predicted.shape)
            lddt_mse = tf.square(lddt-lddt_predicted)
            
            estogram_cost = tf.reduce_mean(estogram_entropy, axis=[0,1,2])
            mask_cost = tf.reduce_mean(mask_entropy, axis=[0,1,2])
            lddt_cost = tf.reduce_mean(lddt_mse)
            
            # Getting the total cost
            cost = self.loss_weight[0]*estogram_cost + self.loss_weight[1]*mask_cost + self.loss_weight[2]*lddt_cost
            
        # Defining optimization procedure.
        # This update_ops thing is necesesary to make batch normalization work.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif self.optimizer == "momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate, 0.8)
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar) for grad, tvar in grads_and_vars]
            train = optimizer.apply_gradients(clipped, name="minimize_cost")

        # Exporting out the operaions as dictionary
        output = dict(obt = obt_in,
                        tbt = tbt_in,
                        idx = idx,
                        val = val,
                        dropout_rate = dropout_rate,
                        isTraining = isTraining,
                        learning_rate = learning_rate,
                        estogram = estogram_in,
                        estogram_predicted = estogram_predicted,
                        mask = mask_in,
                        mask_predicted = mask_predicted,
                        lddt = lddt,
                        lddt_predicted = lddt_predicted,
                        estogram_cost = estogram_cost,
                        mask_cost = mask_cost,
                        lddt_cost = lddt_cost,
                        cost = cost,
                        train = train)
        if self.bert:
            output["bert_in"] = bert_in
        return output
    
    # Calculates LDDT based on estogram
    def calculate_LDDT(self, estogram, mask, center=7):
        with tf.name_scope('lddt'):
            # Remove diagonal from calculation
            mask = tf.multiply(mask, tf.ones(tf.shape(mask))-tf.eye(tf.shape(mask)[0]))
            masked = tf.transpose(tf.multiply(tf.transpose(estogram, [2,0,1]), mask), [1,2,0])

            p0 = tf.reduce_sum(masked[:,:,center], axis=-1)
            p1 = tf.reduce_sum(masked[:,:,center-1]+masked[:,:,center+1], axis=-1)
            p2 = tf.reduce_sum(masked[:,:,center-2]+masked[:,:,center+2], axis=-1)
            p3 = tf.reduce_sum(masked[:,:,center-3]+masked[:,:,center+3], axis=-1)
            p4 = tf.reduce_sum(mask, axis=-1)

            return 0.25 * (4.0*p0 + 3.0*p1 + 2.0*p2 + p3) / p4
    
    def train(self,
              X,
              V,
              epochs,
              decay=0.96,
              base_learning_rate=0.06,
              save_best=True,
              save_freq=-1,
              save_start=-1):

        # Defining the number of batches per epoch.
        start_e = self.e
        while self.e < epochs:
            epoch_cost = []
            
            # Adjust learning rate.
            lr = base_learning_rate*np.power(decay, self.e)
            
            for i in range(len(X.proteins)):
                # Get training data
                f3d, (f1d, bert), f2d, (lddt, estogram, native) = X.next()
                
                # Label smoothing if necessary
                if self.label_smoothing:
                    estogram = apply_label_smoothing(estogram)
                
                feed_dict = {self.ops["obt"]: f1d,\
                             self.ops["tbt"]: f2d,\
                             self.ops["idx"]: f3d[0],\
                             self.ops["val"]: f3d[1],\
                             self.ops["estogram"]: estogram,\
                             self.ops["mask"]: native<15,\
                             self.ops["dropout_rate"]: 0.15,\
                             self.ops["isTraining"]: True,
                             self.ops["learning_rate"]: lr}
                
                if self.bert:
                    feed_dict[self.ops["bert_in"]] = bert
                
                # Get 
                ops_to_run = [self.ops["cost"],\
                              self.ops["estogram_cost"],\
                              self.ops["lddt_cost"],\
                              self.ops["train"]]

                cost, e_cost, l_cost, _ = self.sesh.run(ops_to_run, feed_dict)
                epoch_cost.append(cost)
                sys.stdout.write(("\rEpoch: [%2d/%2d], Batch: [%2d/%2d], loss: %.2f, esto-loss: %.2f, lddt-loss: %.2f :"
                                 %(self.e, epochs, i, len(X.proteins), cost, e_cost, l_cost))+self.name)
                self.b += 1
            self.loss["train"].append(np.mean(epoch_cost))
            
            if V!=None:
                valid_cost = []
                valid_e_cost = []
                valid_l_cost = []
                valid_m_cost = []
                for i in range(len(V.proteins)):
                    f3d, (f1d, bert), f2d, (lddt, estogram, native) = V.next()
                    
                    feed_dict = {self.ops["obt"]: f1d,\
                                 self.ops["tbt"]: f2d,\
                                 self.ops["idx"]: f3d[0],\
                                 self.ops["val"]: f3d[1],\
                                 self.ops["estogram"]: estogram,\
                                 self.ops["mask"]: native<15}            
                    if self.bert:
                        feed_dict[self.ops["bert_in"]] = bert
                    
                    ops_to_run = [self.ops["cost"],\
                                  self.ops["estogram_cost"],\
                                  self.ops["lddt_cost"],\
                                  self.ops["mask_cost"]]

                    cost, e_cost, l_cost, m_cost = self.sesh.run(ops_to_run, feed_dict=feed_dict)
                    valid_cost.append(cost) 
                    valid_e_cost.append(e_cost) 
                    valid_l_cost.append(l_cost) 
                    valid_m_cost.append(m_cost) 
                
                # Take mean over epoch (or 200 samples)
                self.loss["valid"].append(np.mean(valid_cost))
                self.loss["valid_e"].append(np.mean(valid_e_cost))
                self.loss["valid_l"].append(np.mean(valid_l_cost))
                self.loss["valid_m"].append(np.mean(valid_m_cost))
            
            self.e+=1
            
            # Saving the model if needed.
            if self.name != None:
                
                # save_best mode saves a model if it's at first epoch
                # or has better validation score
                if save_best:
                    folder = self.name
                    if V is None:
                        print("\nbest_save option requires validation_set")
                        return
                    
                    if len(self.loss["valid"])==1 or (self.loss["valid"][-1] < np.min(self.loss["valid"][:-1])):
                        self.save(folder, flag=False)
                        f=open(folder+"/README.md", "w")
                        f.write(str(self.e))
                        f.close()
                        
                    # Record loss anyways
                    for k in self.loss.keys():
                        np.save(folder+"/"+k+".npy", np.array(self.loss[k]))
                        
                # save_freq mode
                if save_freq!= -1 and self.e > save_start and self.e%save_freq == 0:
                    folder = "%s_%d" % (self.name, self.e)
                    self.save(folder, flag=False)
                    for k in self.loss.keys():
                        np.save(folder+"/"+k+".npy", np.array(self.loss[k]))
    
    # Predicting given a batch of information
    def predict_and_get_truth(self, batch):
        f3d, (f1d, bert), f2d, dmy = batch
        
        feed_dict = {self.ops["obt"]: f1d,\
                     self.ops["tbt"]: f2d,\
                     self.ops["idx"]: f3d[0],\
                     self.ops["val"]: f3d[1],\
                     self.ops["estogram"]: y[1],\
                     self.ops["mask"]: y[2]<15}
        if self.bert:
            feed_dict[self.ops["bert_in"]] = bert
        
        operations2 = [self.ops["lddt"], 
                       self.ops["estogram"], 
                       self.ops["mask"]]
        
        return (self.predict(batch), self.sesh.run(operations2, feed_dict=feed_dict))
    
    # Predicting given a batch of information
    def predict(self, batch):
        f3d, (f1d, bert), f2d, dmy = batch
        
        feed_dict = {self.ops["obt"]: f1d,\
                     self.ops["tbt"]: f2d,\
                     self.ops["idx"]: f3d[0],\
                     self.ops["val"]: f3d[1]}
        if self.bert:
            feed_dict[self.ops["bert_in"]] = bert
        
        operations = [self.ops["lddt_predicted"], 
                      self.ops["estogram_predicted"], 
                      self.ops["mask_predicted"]]
        
        return self.sesh.run(operations, feed_dict=feed_dict)

    # Saving model
    def save(self, folder, flag=True):
        
        if self.name != None:
            saver = tf.train.Saver(tf.all_variables())
            os.system("mkdir "+folder)
            saver.save(self.sesh, folder+"/model.ckpt")
            
        if flag:
            for k in self.loss.keys():
                np.save(folder+"/"+k+".npy", np.array(self.loss[k]))

    # Functions to load models.
    def load(self, e=-1):
        if e>0:
            self.e = e
            if self.name != None:
                
                # Load at predfined epoch.
                self.verbose: print("Epoch:", e)
                folder = "%s_%d" % (self.name, self.e)
                saver = tf.train.Saver(tf.all_variables())
                saver.restore(self.sesh, folder+"/model.ckpt")
                
                # Read in loss values, truncate it at current e.
                for k in self.loss.keys():
                    self.loss[k] = np.load(folder+"/"+k+".npy").tolist()
        else:
            if self.name != None:
                folder = self.name
                saver = tf.train.Saver(tf.all_variables())
                saver.restore(self.sesh, folder+"/model.ckpt")
                
                # Find which epoch it was
                f=open(folder+"/README.md", "r")
                self.e = int(f.readlines()[0])
                f.close()
                print("Loaded epoch:", self.e)
                    
                # Read in loss values, truncate it at current e.
                for k in self.loss.keys():
                    self.loss[k] = np.load(folder+"/"+k+".npy").tolist()[:self.e]