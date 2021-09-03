'''
    Target model definitions that adverserial examples are attempting to 'fool'

    ref: https://arxiv.org/pdf/1801.02610.pdf
'''
import logging
import os
import tensorflow as tf

import utils



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)



class Target:
    def __init__(self, lr=0.001, epochs=50, n_input=28, n_classes=10, batch_size=256, output_dir='.'):
        self.lr = lr
        self.epochs = epochs
        self.n_input = n_input
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.output_dir = output_dir

    def train_model(self, x_train, y_train, model_name):
        tf.reset_default_graph()

        # define placeholders for input data
        x = tf.placeholder(tf.float32, [None, self.n_input])
        y = tf.placeholder(tf.float32, [None, self.n_classes])
        training = tf.placeholder(tf.bool, [])

        # define computational graph
        logits, _ = self.Model(x, training)

        # define cost
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

        # define optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_name))

        # initialize the variables
        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        # train the target model
        n_batches = len(x_train) // self.batch_size

        for epoch in range(self.epochs):
            avg_cost = 0.

            for i in range(n_batches):
                batch_x, batch_y = utils.next_batch(x_train, y_train, self.batch_size, i)

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, training: True})

                avg_cost += c / n_batches

            print('epoch: %4d, cost: %8.3f' % (epoch + 1, avg_cost))

        # initialize output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # save model
        saver.save(sess, '%s/target_model/%s.ckpt' % (self.output_dir, model_name))
        sess.close()

    def inference_model(self, x_test, y_test, model_name):
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, self.n_input])
        y = tf.placeholder(tf.float32, [None, self.n_classes])
        training = tf.placeholder(tf.bool, shape=())

        logits, probs = self.Model(x, training)

        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_name))
        sess = tf.Session()

        # load model
        saver.restore(sess, '%s/target_model/%s.ckpt' % (self.output_dir, model_name))

        # compute accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        score, prob = sess.run([accuracy, probs], feed_dict={
            x: x_test,
            y: y_test,
            training: False
        })

        print('test accuracy: %0.3f' % (score))
        sess.close()



class Target_A(Target):
    def __init__(self, lr=0.001, epochs=50, n_input=28, n_classes=10, batch_size=256, output_dir='.'):
        Target.__init__(self, lr, epochs, n_input, n_classes, batch_size, output_dir)

    def Model(self, x, training):
        with tf.variable_scope('Model_A', reuse=tf.AUTO_REUSE):
            fc1 = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
            fc2 = tf.layers.dense(inputs=fc1, units=512, activation=tf.nn.relu)
            fc3 = tf.layers.dense(inputs=fc2, units=128, activation=tf.nn.relu)

            logits = tf.layers.dense(inputs=fc3, units=self.n_classes, activation=None)

            probs = tf.nn.softmax(logits)

            return logits, probs

    def train(self, x, y):
        return self.train_model(x, y, 'Model_A')

    def inference(self, x, y):
        return self.inference_model(x, y, 'Model_A')



class Target_B(Target):
    def __init__(self, lr=0.001, epochs=50, n_input=28, n_classes=10, batch_size=256, output_dir='.'):
        Target.__init__(self, lr, epochs, n_input, n_classes, batch_size, output_dir)

    def Model(self, x, training):
        with tf.variable_scope('Model_B', reuse=tf.AUTO_REUSE):
            fc1 = tf.layers.dense(inputs=x, units=1024, activation=None)
            fc1_bn = tf.nn.relu(tf.layers.batch_normalization(fc1, training=training))

            fc2 = tf.layers.dense(inputs=fc1_bn, units=512, activation=None)
            fc2_bn = tf.nn.relu(tf.layers.batch_normalization(fc2, training=training))

            fc3 = tf.layers.dense(inputs=fc2_bn, units=128, activation=None)
            fc3_bn = tf.nn.relu(tf.layers.batch_normalization(fc3, training=training))

            logits = tf.layers.dense(inputs=fc3_bn, units=self.n_classes, activation=None)

            probs = tf.nn.softmax(logits)

            return logits, probs

    def train(self, x, y):
        return self.train_model(x, y, 'Model_B')

    def inference(self, x, y):
        return self.inference_model(x, y, 'Model_B')



class Target_C(Target):
    def __init__(self, lr=0.001, epochs=50, n_input=28, n_classes=10, batch_size=256, output_dir='.'):
        Target.__init__(self, lr, epochs, n_input, n_classes, batch_size, output_dir)

    def Model(self, x, training):
        with tf.variable_scope('Model_C', reuse=tf.AUTO_REUSE):
            fc1 = tf.layers.dense(inputs=x, units=1024, activation=None)
            fc1_bn = tf.nn.relu(tf.layers.batch_normalization(fc1, training=training))

            fc2 = tf.layers.dense(inputs=fc1_bn, units=1024, activation=None)
            fc2_bn = tf.nn.relu(tf.layers.batch_normalization(fc2, training=training))

            fc3 = tf.layers.dense(inputs=fc2_bn, units=512, activation=None)
            fc3_bn = tf.nn.relu(tf.layers.batch_normalization(fc2, training=training))

            fc4 = tf.layers.dense(inputs=fc3_bn, units=512, activation=None)
            fc4_bn = tf.nn.relu(tf.layers.batch_normalization(fc2, training=training))

            fc5 = tf.layers.dense(inputs=fc4_bn, units=128, activation=None)
            fc5_bn = tf.nn.relu(tf.layers.batch_normalization(fc3, training=training))

            logits = tf.layers.dense(inputs=fc5_bn, units=self.n_classes, activation=None)

            probs = tf.nn.softmax(logits)

            return logits, probs

    def train(self, x, y):
        return self.train_model(x, y, 'Model_C')

    def inference(self, x, y):
        return self.inference_model(x, y, 'Model_C')
