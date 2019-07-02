import tensorflow as tf

import os
for file in os.listdir('./results/'):
    if 'ppo_with_gae_model-20' in file:
        print(file)

# delete the current graph
tf.reset_default_graph()

# import the graph from the file
imported_graph = tf.train.import_meta_graph('./results/ppo_with_gae_model-20.meta')

# list all the tensors in the graph
for tensor in tf.get_default_graph().get_operations():
    print (tensor.name)
 
# create saver object
saver = tf.train.Saver()
for i, var in enumerate(saver._var_list):
    print('Var {}: {}'.format(i, var))

# run the session
with tf.Session() as sess:
    # restore the saved vairable
    saver.restore(sess, './results/ppo_with_gae_model-20')
    # print the loaded variable
    policy_ = sess.run(['policy/h1/kernel/Adam:0'])
    print('obs = ', policy_)

