import tensorflow.contrib.slim as slim

# The operation used to print out the configuration
def print_configuration_op(FLAGS):
    print('[Configurations]:')
    #pdb.set_trace()
    for name, value in FLAGS.flag_values_dict().items():
        if type(value) == float:
            print('\t%s: %f'%(name, value))
        elif type(value) == int:
            print('\t%s: %d'%(name, value))
        elif type(value) == str:
            print('\t%s: %s'%(name, value))
        elif type(value) == bool:
            print('\t%s: %s'%(name, value))
        else:
            print('\t%s: %s' % (name, value))

    print('End of configuration')

# VGG19 net
def vgg_19(inputs, scope='vgg_19', reuse=False):
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points
