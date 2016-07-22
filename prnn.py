from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
import random
import time
import six
import sys

class PRNN:
    def __init__(self):
        self.weights = []
        self.input_data = []
        self.output_data = []
        self.seqtl = True
        self.num_data = 0
        self.num_train = []
        self.num_test = []
        self.epochs = None
        self.numhid = 0
        self.arch = []
        self.restored = False
        self.net_name = ""
        self.restored_net = None
        self.input_train = None
        self.input_test = None
        self.output_train = None
        self.output_test = None
        self.best_weights = []
        self.best_biases = []

    def init(self, arch, input_data, output_data, seq = True):

        if (type(arch) == str):
            arch = arch.split("-")
            arch = map(int,arch)
        elif (type(arch) == list):
            arch = map(int,arch)
        else:
            print("Incorrect format of Architecture")

        self.numhid = len(arch)
        self.num_layers = len(arch) + 2

        if len(input_data) != len(output_data):
            print("Input datapoints not equal to output data points")
            return
        else:
            continue

        self.input_data = []
        self.output_data = []

        if not seq:
            data_dict_old = {}
            lent = len(input_data)
            for n in range(lent):
                data_dict_old[input_data[n]] = output_data[n]
            data_dict = {}
            for key in data_dict_old:
                rows = key.shape[0]
                data_dict[key] = data_dict_old[key]
                for n in range(rows):
                    nkey = np.copy(key)
                    np.random.shuffle(nkey)
                    if nkey != key:
                        data_dict[nkey] = data_dict_old[key]
            for key in data_dict:
                self.input_data.append(key)
                self.output_data.append(data_dict[key])
        else:
            self.input_data = input_data
            self.output_data = output_data


        in_lent = self.input_data[0].shape[2] + 1
        out_lent = 1
        self.arch = arch
        self.arch.insert(0,in_lent)
        self.arch.append(out_lent)

        self.num_data = len(self.output_data)
        self.num_train = int(self.num_data*0.8)
        self.num_test = self.num_data - self.num_train
        
        self.input_train = self.input_data[:self.num_train]
        self.input_test = self.input_data[self.num_train:]
        self.output_train = self.output_data[:self.num_train]
        self.output_test = self.output_data[self.num_train:]
        
        self.input_train = np.array(self.input_train, dtype=np.float32)
        self.input_test = np.array(self.input_test, dtype=np.float32)
        self.output_train = np.array(self.output_train, dtype=np.float32)
        self.output_test = np.array(self.output_test, dtype=np.float32)

    def train(self, epochs, log_file = None, net_name = None, dropout = 1.0):
        print("Training...")

        self.epochs = epochs

        if hist_file:
            f = open(hist_file, "w")
            f.close()
        if log_file:
            f = open(log_file + "_test.dat", "w")
            f.close()
            f = open(log_file + "_train.dat", "w")
            f.close()
        with tf.Graph().as_default():
            seed = int(time.time())
            tf.set_random_seed(seed)
            
            input_placeholder = tf.placeholder(tf.float32, shape=(None,self.arch[0]), name = 'input_placeholder')
            output_placeholder = tf.placeholder(tf.float32, shape=(None,self.arch[-1]), name = 'output_placeholder')
            keep_prob = tf.placeholder("float")

            net_out = self.__feed_forward(input_placeholder, keep_prob)

            net_error = self.__errors(net_out,output_placeholder)

            net_train = self.__trainbatch(net_error, Opt, learning_rate)

            #net_eval = self.__errors(net_out, output_placeholder)
            
            sess = tf.Session()
            
            if self.restored:
                saver = tf.train.Saver()
                load_path = "." + self.net_name + "_ckpt"
                saver.restore(sess, load_path)
            else:
                init = tf.initialize_all_variables()
                sess.run(init)

            min_test_rmse = float(100e10)
            for n in range(self.epochs):
                start = 0
                end = 1

                while end < self.num_train:
                    train_batch_input = self.input_train[start:end]
                    train_batch_output = self.output_train[start:end]
                    feed_dict = {input_placeholder: train_batch_input, output_placeholder: train_batch_output, keep_prob: dropout}
                    _,batch_train_rmse = sess.run([net_train, net_error], feed_dict=feed_dict)
                    start = start + 1
                    end = end + 1
                feed_test_dict = {input_placeholder: self.input_test, output_placeholder: self.output_test, keep_prob:1.0}
                test_rmse = (sess.run(net_error, feed_dict=feed_test_dict))**(0.5)
                
                if log_file:
                    feed_train_dict = {input_placeholder: self.input_train, output_placeholder: self.output_train, keep_prob=1.0}
                    train_rmse = (sess.run(net_error, feed_dict=feed_train_dict))**(0.5)
                    f_test = open(log_file + "_test.dat","a")
                    f_test.write(str(n) + " " + str(test_rmse) + "\n")
                    f_train = open(log_file + "_train.dat","a")
                    f_train.write(str(n) + " " + str(train_rmse) + "\n")
                    f_test.close()
                    f_train.close()

                if (test_rmse < min_test_rmse):
                    if net_name:
                        save_params = tf.train.Saver()
                        save_params.save(sess,"." + net_name + "_ckpt")
                        f_net = open("." + net_name + "_config", "w")
                        f_net.write(str(self.arch[1:-1]).replace("[","").replace("]",""))
                        f_net.close()
                    if len(self.best_weights) > 0:
                        del self.best_weights[:]
                        del self.best_biases[:]
                    for k in range(len(self.arch)-1):
                        self.best_weights.append(self.weights[k])
                        self.best_biases.append(self.biases[k])
                    min_test_rmse = test_rmse
                #start_testcode
                #print shp
                #print("Epoch number: " + str(n) + "\r",)
                self.__printProgress(n+1,self.epochs)
                #print("The current train rmse is: ", train_rmse)
                #print("The current test rmse is: ", test_rmse)
                #print("\r")
                #start_testcode
            print("Training complete!")
            if hist_file:
                net_best_out = self.__best_activations(sess)
                net_hist_errors = self.__get_hist_errors(net_best_out, self.output_test, sess)
                self.__histtofile(hist_file, net_hist_errors)

    def restore(self, net_name, input_file = ""):
        self.restored = True
        self.net_name = net_name
        if input_file != "":
            f_net = open("." + net_name + "_config", "r")
            arch = f_net.readline().split(",")
            for n in range(len(arch)):
                arch[n] = int(arch[n])
            f_net.close()
            self.init(arch, input_file)
            return
        else:
            return
        
    def __printProgress(self, iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
        """
        Call in a loop to create terminal progress bar
        @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
        """
        filledLength    = int(round(barLength * iteration / float(total)))
        percents        = round(100.00 * (iteration / float(total)), decimals)
        bar             = '#' * filledLength + '-' * (barLength - filledLength)
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
        sys.stdout.flush()
        if iteration == total:
            sys.stdout.write('\n')
            sys.stdout.flush()
            return

    def __feed_forward(self, inputs, keep_prob):
        for n in range(self.num_layers - 1):
            weight_name = "weights" + str(n)
            bias_name = "biases" + str(n)
            self.weights.append(tf.Variable(tf.truncated_normal([self.arch[n],self.arch[n+1]], dtype=np.float32), name=weight_name))
            self.biases.append(tf.Variable(tf.truncated_normal([self.arch[n+1]], dtype=np.float32), name=bias_name))

        shp = tf.shape(inputs)
        inputs = tf.split(0,shp[0],inputs)
        temp = [0]
        temp_active = 0
        for k in range(len(inputs)):
            inputs[k] = tf.concat(0,[temp_active, inputs[k]])
            activations = inputs[k]
            for n in range(self.num_layers - 1):
                activations = tf.add(tf.nn.tanh(tf.matmul(activations,self.weights[n]) + self.biases[n]), 0.1*(tf.matmul(activations,self.weights[n]) + self.biases[n]))
                if n != (self.num_layers - 2):
                    activations = tf.nn.dropout(activations, keep_prob)
            temp_active = activations
            temp.append(temp_active)
        return sum(temp)

    def __errors(self,pred_output, act_output):
        if self.classreg == 1:
            diff = tf.sub(pred_output, act_output)
            errors = tf.reduce_mean(tf.reduce_sum(tf.mul(diff,diff), 1), 0)
        return errors

    def __trainbatch(self, errors, Opt, learning_rate = 0.0):
        if Opt == "Adam":
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(errors)
        return train_op

    def __best_activations(self,sess):
        activations = self.input_test
        if self.classreg == 1:
            for n in range(self.num_layers - 1):
                activations = tf.add(tf.nn.tanh(tf.matmul(activations,self.best_weights[n]) + self.best_biases[n]), 0.1*(tf.matmul(activations,self.best_weights[n]) + self.best_biases[n])).eval(session=sess)
        return activations

    
    def __get_hist_errors(self, pred_output, act_output, sess):
        return tf.sub(pred_output, act_output).eval(session=sess)

    
    def __histtofile(self, hist_file, errors):
        f = open(hist_file, "w")
        for n in errors:
            f.write(str(n[0]) + "\n")
        return

            
    def __fin_parser(self, strng, sep):
        l1 = strng.split(sep)
        temp = []
        for n in l1:
            temp.append(float(n))
        return temp

    def __reset(self,meth_name):
        if (meth_name == "init"):
            self.weights = []
            self.biases = []
            self.best_weights = []
            self.best_biases = []
            self.input_data = []
            self.input_train = []
            self.input_test = []
            self.output_data = []
            self.output_train = []
            self.output_test = []
            self.num_data = 0
            self.num_train = 0
            self.num_test = 0
            self.epochs = 100
            self.numhid = 0
            self.arch = []
            

