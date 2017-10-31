import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

class FCNSegementer(object):

    '''
    Constructor for setting params
    '''
    def __init__(self, params):
        for p in params:
            setattr(self, p, params[p])

    """
        Load Pretrained VGG Model into TensorFlow.
        :param sess: TensorFlow Session
        :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
        :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    def load_vgg(self, sess, vgg_path):
        vgg_tag = 'vgg16'
        vgg_input_tensor_name = 'image_input:0'
        vgg_keep_prob_tensor_name = 'keep_prob:0'
        vgg_layer3_out_tensor_name = 'layer3_out:0'
        vgg_layer4_out_tensor_name = 'layer4_out:0'
        vgg_layer7_out_tensor_name = 'layer7_out:0'

        #   Use tf.saved_model.loader.load to load the model and weights
        tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

        default_graph = tf.get_default_graph()
        vgg_image_input = default_graph.get_tensor_by_name(vgg_input_tensor_name)
        vgg_keep = default_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        vgg_layer3 = default_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        vgg_layer4 = default_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        vgg_layer7 = default_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
        
        return vgg_image_input, vgg_keep, vgg_layer3, vgg_layer4, vgg_layer7

    def save_model(self, sess):
        model_file = os.path.join(self.logs_location, "model")
        saver = tf.train.Saver()
        saver.save(sess, model_file)
        tf.train.write_graph(sess.graph_def, self.logs_location, "model.pb", False)
        print("Model saved")

    def layers(self, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
        def conv_1_by_1(x, num_classes, 
                        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                        init = tf.truncated_normal_initializer(stddev = 0.01)):
            return tf.layers.conv2d(x, num_classes, 1,1, padding = 'same', kernel_regularizer =  kernel_regularizer, kernel_initializer = init)

        def upsample(x, num_classes, kernel_size, strides, 
                        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                        init = tf.truncated_normal_initializer(stddev = 0.01)):
            return tf.layers.conv2d_transpose(x, num_classes, kernel_size, strides, padding = 'same', kernel_regularizer = kernel_regularizer, kernel_initializer = init)

        l7_1x1 = conv_1_by_1(vgg_layer7_out, num_classes)
        l4_1x1 = conv_1_by_1(vgg_layer4_out, num_classes)
        l3_1x1 = conv_1_by_1(vgg_layer3_out, num_classes)
    
        #upsample l7 by 2
        l7_upsample = upsample(l7_1x1, num_classes, 4, 2)
        #l7_upsample = tf.layers.batch_normalization(l7_upsample)
    
        #add skip connection from  l4_1x1    
        l7l4_skip = tf.add(l7_upsample, l4_1x1)

        #implement the another transposed convolution layer
        l7l4_upsample = upsample(l7l4_skip, num_classes, 4, 2)
        #l7l4_upsample = tf.layers.batch_normalization(l7l4_upsample)

        #add second skip connection from l3_1x1        
        l7l4l3_skip = tf.add(l7l4_upsample, l3_1x1)
    
        return upsample(l7l4l3_skip, num_classes, 16, 8)

    """
        Build the TensorFLow loss and optimizer operations.
        :param nn_last_layer: TF Tensor of the last layer in the neural network
        :param correct_label: TF Placeholder for the correct label image
        :param learning_rate: TF Placeholder for the learning rate
        :param num_classes: Number of classes to classify
        :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    def optimize(self, nn_last_layer, correct_label, learning_rate, num_classes):
        logits = tf.reshape(nn_last_layer, (-1, num_classes))
        correct_label = tf.reshape(correct_label, (-1, num_classes))
    
        # define a loss function and a trainer/optimizer    
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return logits, optimizer, loss


    """
        Train neural network and print out the loss during training.
        :param sess: TF Session
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
        :param train_op: TF Operation to train the neural network
        :param cross_entropy_loss: TF Tensor for the amount of loss
        :param input_image: TF Placeholder for input images
        :param correct_label: TF Placeholder for label images
        :param keep_prob: TF Placeholder for dropout keep probability
        :param learning_rate: TF Placeholder for learning rate
    """
    def train_nn(self, sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
        for epoch in range(epochs):
            # train on batches        
            for images, labels in get_batches_fn(batch_size):
                _, loss = sess.run([train_op, cross_entropy_loss],
                feed_dict={input_image: images, 
                            correct_label: labels, 
                            keep_prob:self.keep_prob, 
                            learning_rate:self.lr})

                print("Epoch {} of {}...".format(epoch+1, epochs), "Training Loss: {:.5f}...".format(loss))

    '''
    Run tests
    '''
    def run(self):
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.allocator_type = 'BFC'
        # Download pretrained vgg model
        helper.maybe_download_pretrained_vgg(self.data_dir)
        
        # Path to vgg model and training data
        vgg_path = os.path.join(self.data_dir, 'vgg')
        train_path = os.path.join(self.data_dir, self.training_dir)
        
        # Generate batches
        get_batches_fn = helper.gen_batch_function(train_path, self.image_shape)

        with tf.Session() as sess:
            correct_label = tf.placeholder(tf.float32, [None, None, None, self.num_classes])
            learning_rate = tf.placeholder(tf.float32)
            
                        
            # Build FCN using load_vgg, layers
            vgg_image_input, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = self.load_vgg(sess, vgg_path)
            nn_last_layer = self.layers(vgg_layer3, vgg_layer4, vgg_layer7, self.num_classes)

             # Optimise cross entropy loss
            logits, train_op, cross_entropy_loss = self.optimize(nn_last_layer, correct_label, learning_rate, self.num_classes)
            
            # Train NN 
            sess.run(tf.global_variables_initializer())        
            self.train_nn(sess, self.epochs, self.batch_size, get_batches_fn, train_op, cross_entropy_loss, vgg_image_input,
                 correct_label, keep_prob, learning_rate)

            #save the model 
            self.save_model(sess)

            # Save inference data u
            helper.save_inference_samples(self.runs_dir, self.data_dir, sess, self.image_shape, logits, keep_prob, vgg_image_input)

    '''
    Run tests
    '''
    def run_tests(self):
        tests.test_load_vgg(self.load_vgg, tf)
        tests.test_layers(self.layers)
        tests.test_optimize(self.optimize_cross_entropy)
        tests.test_train_nn(self.train_nn)

if __name__ == '__main__':

    # training hyper parameters
    params = {
        'data_dir':        'data',
        'runs_dir':        'runs',
        'training_dir':    'data_road/training',
        'logs_location':   'logs',
        'lr':              0.0001,
        'keep_prob':       0.25,
        'epochs':          25,
        'batch_size':      16,
        'std_init':        0.01,
        'num_classes':     2,
        'image_shape':     (160, 576)
    }
    fcn = FCNSegementer(params)
    fcn.run()
