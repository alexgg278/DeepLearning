import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

ATT_FILE = "MedianHouseValuePreparedCleanAttributes.csv"
LABEL_FILE = "MedianHouseValueOneHotEncodedClasses.csv"


TRAIN_RATE = 0.8

attributes = pd.read_csv(ATT_FILE)
label = pd.read_csv(LABEL_FILE)

n_instances = attributes.shape[0]
n_train = int(n_instances*TRAIN_RATE)
n_dev = int((n_instances-n_train)/2)
n_test = n_instances-n_train-n_dev

x_train = attributes.values[:n_train]
t_train = label.values[:n_train]

x_dev = attributes.values[n_train:n_train+n_dev]
t_dev = label.values[n_train:n_train+n_dev]

x_test = attributes.values[n_train+n_dev:n_instances]
t_test = label.values[n_train+n_dev:n_instances]

INPUTS = x_train.shape[1]
OUTPUTS = t_train.shape[1]
NUM_TRAINING_EXAMPLES = int(round(x_train.shape[0]/1))
NUM_DEV_EXAMPLES = int (round (x_dev.shape[0]/1))
NUM_TEST_EXAMPLES = int (round (x_test.shape[0]/1))



n_epochs = 15000
batch_size = 128
n_neurons_per_layer = [360,240,200,160,120,100,80,40,30,10] 

X = tf.placeholder (dtype=tf.float32, shape=(None,INPUTS),name="X")
t = tf.placeholder (dtype=tf.float32, shape=(None,OUTPUTS), name="t")

hidden_layers = []
hidden_layers.append(tf.layers.dense (X, n_neurons_per_layer[0], 
                                      activation = tf.nn.relu))
for layer in range(1,len(n_neurons_per_layer)):
    hidden_layers.append(tf.layers.dense (hidden_layers[layer-1], 
                 n_neurons_per_layer[layer], activation = tf.nn.relu))
net_out = tf.layers.dense (hidden_layers[len(n_neurons_per_layer)-1], OUTPUTS)
y = tf.nn.softmax (logits=net_out, name="y")

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2 (labels=t, logits=net_out)
mean_log_loss = tf.reduce_mean (cross_entropy, name="cost")

correct_predictions = tf.equal(tf.argmax(y,1),tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))

initial_learning_rate = 0.1
decay_steps = 1000
decay_rate = 0.96
global_step = tf.Variable(0, trainable=False, name="global_step")
learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step, decay_steps,decay_rate)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate)#.minimize(mean_log_loss)
#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9,use_nesterov=True)
#optimizer = tf.train.RMSPropOptimizer(learning_rate= 0.1,decay=0.9, epsilon=1e-10)
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(mean_log_loss,global_step=global_step)

init = tf.global_variables_initializer()
accuracy_train_history = []
with tf.Session() as sess:
    sess.run(init)
    for epoch in tqdm(range(n_epochs)):
        offset = (epoch * batch_size) % (NUM_TRAINING_EXAMPLES - batch_size)
        sess.run (train_step, feed_dict={X: x_train[offset:(offset+batch_size)],
                                         t: t_train[offset:(offset+batch_size)]})
            
    accuracy_test = accuracy.eval(feed_dict={X: x_test[:NUM_TEST_EXAMPLES],
                                              t: t_test[:NUM_TEST_EXAMPLES]})
    test_predictions = y.eval(feed_dict={X: x_test[:NUM_TEST_EXAMPLES]})
    
    test_correct_preditions = correct_predictions.eval (feed_dict=
                                    {X: x_test[:NUM_TEST_EXAMPLES],
                                     t: t_test[:NUM_TEST_EXAMPLES]}
    )   

print("Accuracy for the TEST set: " + str(accuracy_test))