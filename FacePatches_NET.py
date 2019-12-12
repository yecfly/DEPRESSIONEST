from NETWORK import Network
import tflearn
import tensorflow as tf

#3
class FacePatches_NET_3Conv_2Inception(Network):
    def setup(self):
        (self.feed('eyePatch_data')#64*26
             .conv(3, 3, 8, 1, 1, name='eye_conv1_1_3x3')#64*26*16
             .conv(3, 3, 8, 1, 1, name='eye_conv1_2_3x3')#64*26*16
             .max_pool(2, 2, 2, 2, name='eye_pool1')#32*13*16
             .conv(3, 3, 32, 1, 1, name='eye_conv2_1_3x3')#32*13*64
             .conv(3, 3, 32, 1, 1, name='eye_conv2_2_3x3')#
             .max_pool(2, 2, 2, 2, name='eye_pool2')
             .conv(3, 3, 128, 1, 1, name = 'eye_conv3_1_3x3')
             .conv(3, 3, 128, 1, 1, name = 'eye_conv3_2_3x3')
             .max_pool(2, 2, 2, 2, name = 'eye_pool3')
             .fc(1024, name='eye_fc1'))
        (self.feed('eye_pool2')
             .fc(1024, name = 'eye_fc2'))
          

        (self.feed('middlePatch_data')#28*49
             .conv(3, 3, 8, 1, 1, name='middle_conv1_1_3x3')#
             .conv(3, 3, 8, 1, 1, name='middle_conv1_2_3x3')#
             .max_pool(2, 2, 2, 2, name='middle_pool1')#
             .conv(3, 3, 32, 1, 1, name='middle_conv2_1_3x3')
             .conv(3, 3, 32, 1, 1, name='middle_conv2_2_3x3')
             .max_pool(2, 2, 2, 2, name='middle_pool2')
             .conv(3, 3, 128, 1, 1, name = 'middle_conv3_1_3x3')
             .conv(3, 3, 128, 1, 1, name = 'middle_conv3_2_3x3')
             .max_pool(2, 2, 2, 2, name = 'middle_pool3')
             .fc(1024, name='middle_fc1'))#
        (self.feed('middle_pool2')
             .fc(1024, name = 'middle_fc2'))

        (self.feed('mouthPatch_data')#54*30
             .conv(3, 3, 8, 1, 1, name='mouth_conv1_1_3x3')#
             .conv(3, 3, 8, 1, 1, name='mouth_conv1_2_3x3')#
             .max_pool(2, 2, 2, 2, name='mouth_pool1')#
             .conv(3, 3, 32, 1, 1, name='mouth_conv2_1_3x3')
             .conv(3, 3, 32, 1, 1, name='mouth_conv2_2_3x3')
             .max_pool(2, 2, 2, 2, name='mouth_pool2')
             .conv(3, 3, 128, 1, 1, name = 'mouth_conv3_1_3x3')
             .conv(3, 3, 128, 1, 1, name = 'mouth_conv3_2_3x3')
             .max_pool(2, 2, 2, 2, name = 'mouth_pool3')
             .fc(1024, name='mouth_fc1'))#
        (self.feed('mouth_pool2')
             .fc(1024, name = 'mouth_fc2'))

        (self.feed('eye_fc1',
                   'eye_fc2',
                   'middle_fc1',
                   'middle_fc2',
                   'mouth_fc1',
                   'mouth_fc2')
             .concat(1, name='fusion_1')
             .fc(2048, name='fc1')
             .dropout(0.5, name='drop1')
             .fc(2048, name = 'fc2')
             .dropout(0.5, name='drop2')
             .fc(7, relu = False, name = 'fc7')
             .softmax(name = 'prob'))
###using net 5 instead
def FacePatches_NET_3Conv_2Inception_tflearn(eyep, middlep, mouthp, classNo=7):
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')
    e_net=tf.concat([e_net, efc2], 1, name='eye_fc')

    mi_net=tflearn.conv_2d(middlep, 8, 3, activation='relu',name='middle_conv1_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 8, 3, activation='relu',name='middle_conv1_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net,2,2,name='middle_pool1')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool2')
    mifc2 = tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc2')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool3')
    mi_net=tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc1')
    mi_net=tf.concat([mi_net, mifc2], 1, name='middle_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')
    mo_net=tf.concat([mo_net, mfc2], 1, name='mouth_fc')

    fc_net=tf.concat([e_net,mi_net,mo_net], 1, name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, classNo, activation='softmax', name='prob')
    return softmax

#4
class FacePatches_NET_3Conv_1Inception(Network):
    def setup(self):
        (self.feed('eyePatch_data')#64*26
             .conv(3, 3, 8, 1, 1, name='eye_conv1_1_3x3')#64*26*16
             .conv(3, 3, 8, 1, 1, name='eye_conv1_2_3x3')#64*26*16
             .max_pool(2, 2, 2, 2, name='eye_pool1')#32*13*16
             .conv(3, 3, 32, 1, 1, name='eye_conv2_1_3x3')#32*13*64
             .conv(3, 3, 32, 1, 1, name='eye_conv2_2_3x3')#
             .max_pool(2, 2, 2, 2, name='eye_pool2')
             .conv(3, 3, 128, 1, 1, name = 'eye_conv3_1_3x3')
             .conv(3, 3, 128, 1, 1, name = 'eye_conv3_2_3x3')
             .max_pool(2, 2, 2, 2, name = 'eye_pool3')
             .fc(1024, name='eye_fc1'))
        #(self.feed('eye_pool2')
        #     .fc(1024, name = 'eye_fc2'))
          

        (self.feed('middlePatch_data')#28*49
             .conv(3, 3, 8, 1, 1, name='middle_conv1_1_3x3')#
             .conv(3, 3, 8, 1, 1, name='middle_conv1_2_3x3')#
             .max_pool(2, 2, 2, 2, name='middle_pool1')#
             .conv(3, 3, 32, 1, 1, name='middle_conv2_1_3x3')
             .conv(3, 3, 32, 1, 1, name='middle_conv2_2_3x3')
             .max_pool(2, 2, 2, 2, name='middle_pool2')
             .conv(3, 3, 128, 1, 1, name = 'middle_conv3_1_3x3')
             .conv(3, 3, 128, 1, 1, name = 'middle_conv3_2_3x3')
             .max_pool(2, 2, 2, 2, name = 'middle_pool3')
             .fc(1024, name='middle_fc1'))#
        #(self.feed('middle_pool2')
        #     .fc(1024, name = 'middle_fc2'))

        (self.feed('mouthPatch_data')#54*30
             .conv(3, 3, 8, 1, 1, name='mouth_conv1_1_3x3')#
             .conv(3, 3, 8, 1, 1, name='mouth_conv1_2_3x3')#
             .max_pool(2, 2, 2, 2, name='mouth_pool1')#
             .conv(3, 3, 32, 1, 1, name='mouth_conv2_1_3x3')
             .conv(3, 3, 32, 1, 1, name='mouth_conv2_2_3x3')
             .max_pool(2, 2, 2, 2, name='mouth_pool2')
             .conv(3, 3, 128, 1, 1, name = 'mouth_conv3_1_3x3')
             .conv(3, 3, 128, 1, 1, name = 'mouth_conv3_2_3x3')
             .max_pool(2, 2, 2, 2, name = 'mouth_pool3')
             .fc(1024, name='mouth_fc1'))#
        #(self.feed('mouth_pool2')
        #     .fc(1024, name = 'mouth_fc2'))



        (self.feed('eye_fc1',
                   'middle_fc1',
                   'mouth_fc1')
             .concat(1, name='fusion_1')
             .fc(2048, name='fc1')
             .dropout(0.5, name='drop1')
             .fc(2048, name = 'fc2')
             .dropout(0.5, name='drop2')
             .fc(7, relu = False, name = 'fc7')
             .softmax(name = 'prob'))

#4 nv3
def FacePatches_NET_3Conv_IInception_tflear(eyep, middlep, mouthp, classNo=7):
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')

    mi_net=tflearn.conv_2d(middlep, 8, 3, activation='relu',name='middle_conv1_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 8, 3, activation='relu',name='middle_conv1_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net,2,2,name='middle_pool1')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool2')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool3')
    mi_net=tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc1')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')

    fc_net=tf.concat([e_net,mi_net,mo_net], 1, name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, classNo, activation='softmax', name='prob')
    return softmax

###using net 5 linear on the final fullyconnected layer
def FacePatches_NET_3Conv_3Inception_tflearn_5(eyep, middlep, mouthp, classNo=7):
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    efc3 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')
    #e_net=tf.concat([e_net, efc2, efc3], 1, name='eye_fc')#3072
    e_net=tflearn.layers.merge([e_net, efc2, efc3], 'concat', name='eye_fc')

    mi_net=tflearn.conv_2d(middlep, 8, 3, activation='relu',name='middle_conv1_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 8, 3, activation='relu',name='middle_conv1_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net,2,2,name='middle_pool1')
    mifc3 = tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool2')
    mifc2 = tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc2')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool3')
    mi_net=tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc1')
    #mi_net=tf.concat([mi_net, mifc2, mifc3], 1, name='middle_fc')#3072
    mi_net=tflearn.layers.merge([mi_net, mifc2, mifc3], 'concat', name='middle_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mfc3 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')
    #mo_net=tf.concat([mo_net, mfc2, mfc3], 1, name='mouth_fc')#3072
    mo_net=tflearn.layers.merge([mo_net, mfc2, mfc3], 'concat', name='mouth_fc')

    #fc_net=tf.concat([e_net,mi_net,mo_net], 1, name='fusion_1')#9216
    fc_net=tflearn.layers.merge([e_net,mi_net,mo_net], 'concat', name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, classNo, name='prob')
    return softmax
###using net 6
def FacePatches_NET_3Conv_3Inception_tflearn(eyep, middlep, mouthp, classNo=7):
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    efc3 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')
    #e_net=tf.concat([e_net, efc2, efc3], 1, name='eye_fc')#3072
    e_net=tflearn.layers.merge([e_net, efc2, efc3], 'concat', name='eye_fc')

    mi_net=tflearn.conv_2d(middlep, 8, 3, activation='relu',name='middle_conv1_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 8, 3, activation='relu',name='middle_conv1_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net,2,2,name='middle_pool1')
    mifc3 = tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool2')
    mifc2 = tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc2')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool3')
    mi_net=tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc1')
    #mi_net=tf.concat([mi_net, mifc2, mifc3], 1, name='middle_fc')#3072
    mi_net=tflearn.layers.merge([mi_net, mifc2, mifc3], 'concat', name='middle_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mfc3 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')
    #mo_net=tf.concat([mo_net, mfc2, mfc3], 1, name='mouth_fc')#3072
    mo_net=tflearn.layers.merge([mo_net, mfc2, mfc3], 'concat', name='mouth_fc')

    #fc_net=tf.concat([e_net,mi_net,mo_net], 1, name='fusion_1')#9216
    fc_net=tflearn.layers.merge([e_net,mi_net,mo_net], 'concat', name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, classNo, activation='softmax', name='prob')
    return softmax

###using net 7
def FacePatches_NET_3Conv_3Inception_tflearn_ELU(eyep, middlep, mouthp, classNo=7):
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    efc3 = tflearn.fully_connected(e_net, 1536, activation='tanh', name='eye_fc3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.fully_connected(e_net, 1536, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.avg_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1536, activation='tanh', name='eye_fc1')
    #e_net=tf.concat([e_net, efc2, efc3], 1, name='eye_fc')#3072
    e_net=tflearn.layers.merge([e_net, efc2, efc3], 'concat', name='eye_fc')

    mi_net=tflearn.conv_2d(middlep, 8, 3, activation='relu',name='middle_conv1_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 8, 3, activation='relu',name='middle_conv1_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net,2,2,name='middle_pool1')
    mifc3 = tflearn.fully_connected(mi_net, 1536, activation='tanh', name='middle_fc3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool2')
    mifc2 = tflearn.fully_connected(mi_net, 1536, activation='tanh', name='middle_fc2')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_2_3x3')
    mi_net=tflearn.avg_pool_2d(mi_net, 2, 2, name='middle_pool3')
    mi_net=tflearn.fully_connected(mi_net, 1536, activation='tanh', name='middle_fc1')
    #mi_net=tf.concat([mi_net, mifc2, mifc3], 1, name='middle_fc')#3072
    mi_net=tflearn.layers.merge([mi_net, mifc2, mifc3], 'concat', name='middle_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mfc3 = tflearn.fully_connected(mo_net, 1536, activation='tanh', name='mouth_fc3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.fully_connected(mo_net, 1536, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.avg_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1536, activation='tanh', name='mouth_fc1')
    #mo_net=tf.concat([mo_net, mfc2, mfc3], 1, name='mouth_fc')#3072
    mo_net=tflearn.layers.merge([mo_net, mfc2, mfc3], 'concat', name='mouth_fc')

    #fc_net=tf.concat([e_net,mi_net,mo_net], 1, name='fusion_1')#9216
    fc_net=tflearn.layers.merge([e_net,mi_net,mo_net], 'concat', name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 3072, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.6, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 3072, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.6, name='drop2')
    softmax=tflearn.fully_connected(fc_net, classNo, activation='softmax', name='prob')
    return softmax

###using net 8
def FacePatches_NET_3Conv_3Inception_tflearn_8(eyep, middlep, mouthp, classNo=7):
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    efc3 = tflearn.fully_connected(e_net, 512, activation='tanh', name='eye_fc3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.fully_connected(e_net, 512, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.avg_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 512, activation='tanh', name='eye_fc1')
    #e_net=tf.concat([e_net, efc2, efc3], 1, name='eye_fc')#3072
    e_net=tflearn.layers.merge([e_net, efc2, efc3], 'concat', name='eye_fc')

    mi_net=tflearn.conv_2d(middlep, 8, 3, activation='relu',name='middle_conv1_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 8, 3, activation='relu',name='middle_conv1_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net,2,2,name='middle_pool1')
    mifc3 = tflearn.fully_connected(mi_net, 512, activation='tanh', name='middle_fc3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool2')
    mifc2 = tflearn.fully_connected(mi_net, 512, activation='tanh', name='middle_fc2')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_2_3x3')
    mi_net=tflearn.avg_pool_2d(mi_net, 2, 2, name='middle_pool3')
    mi_net=tflearn.fully_connected(mi_net, 512, activation='tanh', name='middle_fc1')
    #mi_net=tf.concat([mi_net, mifc2, mifc3], 1, name='middle_fc')#3072
    mi_net=tflearn.layers.merge([mi_net, mifc2, mifc3], 'concat', name='middle_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mfc3 = tflearn.fully_connected(mo_net, 512, activation='tanh', name='mouth_fc3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.fully_connected(mo_net, 512, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.avg_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 512, activation='tanh', name='mouth_fc1')
    #mo_net=tf.concat([mo_net, mfc2, mfc3], 1, name='mouth_fc')#3072
    mo_net=tflearn.layers.merge([mo_net, mfc2, mfc3], 'concat', name='mouth_fc')

    #fc_net=tf.concat([e_net,mi_net,mo_net], 1, name='fusion_1')#9216
    fc_net=tflearn.layers.merge([e_net,mi_net,mo_net], 'concat', name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.7, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.7, name='drop2')
    softmax=tflearn.fully_connected(fc_net, classNo, activation='softmax', name='prob')
    return softmax

###using net 9
def FacePatches_NET_3Conv_3Inception_tflearn_9(eyep, middlep, mouthp, classNo=7):
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    efc3 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.avg_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')
    #e_net=tf.concat([e_net, efc2, efc3], 1, name='eye_fc')#3072
    e_net=tflearn.layers.merge([e_net, efc2, efc3], 'concat', name='eye_fc')

    mi_net=tflearn.conv_2d(middlep, 8, 3, activation='relu',name='middle_conv1_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 8, 3, activation='relu',name='middle_conv1_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net,2,2,name='middle_pool1')
    mifc3 = tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool2')
    mifc2 = tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc2')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_2_3x3')
    mi_net=tflearn.avg_pool_2d(mi_net, 2, 2, name='middle_pool3')
    mi_net=tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc1')
    #mi_net=tf.concat([mi_net, mifc2, mifc3], 1, name='middle_fc')#3072
    mi_net=tflearn.layers.merge([mi_net, mifc2, mifc3], 'concat', name='middle_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mfc3 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.avg_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')
    #mo_net=tf.concat([mo_net, mfc2, mfc3], 1, name='mouth_fc')#3072
    mo_net=tflearn.layers.merge([mo_net, mfc2, mfc3], 'concat', name='mouth_fc')

    #fc_net=tf.concat([e_net,mi_net,mo_net], 1, name='fusion_1')#9216
    fc_net=tflearn.layers.merge([e_net,mi_net,mo_net], 'concat', name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, classNo, activation='softmax', name='prob')
    return softmax

###using net 10
def FacePatches_NET_3Conv_3Inception_tflearn_10(eyep, middlep, mouthp, classNo=7):
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    efc3 = tflearn.conv_2d(e_net, 1, 1, activation='tanh', name='eye_downsampling1')
    efc3 = tflearn.fully_connected(efc3, 1024, activation='tanh', name='eye_fc3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.conv_2d(e_net, 1, 1, activation='tanh', name='eye_downsampling2')
    efc2 = tflearn.fully_connected(efc2, 1024, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net = tflearn.conv_2d(e_net, 1, 1, activation='tanh', name='eye_downsampling3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')
    #e_net=tf.concat([e_net, efc2, efc3], 1, name='eye_fc')#3072
    e_net=tflearn.layers.merge([e_net, efc2, efc3], 'concat', name='eye_fc')

    mi_net=tflearn.conv_2d(middlep, 8, 3, activation='relu',name='middle_conv1_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 8, 3, activation='relu',name='middle_conv1_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net,2,2,name='middle_pool1')
    mifc3 = tflearn.conv_2d(mi_net, 1, 1, activation='tanh', name='middle_downsampling1')
    mifc3 = tflearn.fully_connected(mifc3, 1024, activation='tanh', name='middle_fc3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool2')
    mifc2 = tflearn.conv_2d(mi_net, 1, 1, activation='tanh', name='middle_downsampling2')
    mifc2 = tflearn.fully_connected(mifc2, 1024, activation='tanh', name='middle_fc2')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool3')
    mi_net = tflearn.conv_2d(mi_net, 1, 1, activation='tanh', name='middle_downsampling3')
    mi_net=tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc1')
    #mi_net=tf.concat([mi_net, mifc2, mifc3], 1, name='middle_fc')#3072
    mi_net=tflearn.layers.merge([mi_net, mifc2, mifc3], 'concat', name='middle_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mfc3 = tflearn.conv_2d(mo_net, 1, 1, activation='tanh', name='mouth_downsampling1')
    mfc3 = tflearn.fully_connected(mfc3, 1024, activation='tanh', name='mouth_fc3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.conv_2d(mo_net, 1, 1, activation='tanh', name='mouth_downsampling2')
    mfc2 = tflearn.fully_connected(mfc2, 1024, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net = tflearn.conv_2d(mo_net, 1, 1, activation='tanh', name='mouth_downsampling3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')
    #mo_net=tf.concat([mo_net, mfc2, mfc3], 1, name='mouth_fc')#3072
    mo_net=tflearn.layers.merge([mo_net, mfc2, mfc3], 'concat', name='mouth_fc')

    #fc_net=tf.concat([e_net,mi_net,mo_net], 1, name='fusion_1')#9216
    fc_net=tflearn.layers.merge([e_net,mi_net,mo_net], 'concat', name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, classNo, activation='softmax', name='prob')
    return softmax

###using net 11
def FacePatches_NET_3Conv_3Inception_tflearn_11(eyep, middlep, mouthp, classNo=7):
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    efc3 = tflearn.conv_2d(e_net, 1, 4, activation='tanh', name='eye_downsampling1')
    efc3 = tflearn.fully_connected(efc3, 1024, activation='tanh', name='eye_fc3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.conv_2d(e_net, 1, 4, activation='tanh', name='eye_downsampling2')
    efc2 = tflearn.fully_connected(efc2, 1024, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net = tflearn.conv_2d(e_net, 1, 4, activation='tanh', name='eye_downsampling3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')
    #e_net=tf.concat([e_net, efc2, efc3], 1, name='eye_fc')#3072
    e_net=tflearn.layers.merge([e_net, efc2, efc3], 'concat', name='eye_fc')

    mi_net=tflearn.conv_2d(middlep, 8, 3, activation='relu',name='middle_conv1_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 8, 3, activation='relu',name='middle_conv1_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net,2,2,name='middle_pool1')
    mifc3 = tflearn.conv_2d(mi_net, 1, 4, activation='tanh', name='middle_downsampling1')
    mifc3 = tflearn.fully_connected(mifc3, 1024, activation='tanh', name='middle_fc3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool2')
    mifc2 = tflearn.conv_2d(mi_net, 1, 4, activation='tanh', name='middle_downsampling2')
    mifc2 = tflearn.fully_connected(mifc2, 1024, activation='tanh', name='middle_fc2')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool3')
    mi_net = tflearn.conv_2d(mi_net, 1, 4, activation='tanh', name='middle_downsampling3')
    mi_net=tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc1')
    #mi_net=tf.concat([mi_net, mifc2, mifc3], 1, name='middle_fc')#3072
    mi_net=tflearn.layers.merge([mi_net, mifc2, mifc3], 'concat', name='middle_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mfc3 = tflearn.conv_2d(mo_net, 1, 4, activation='tanh', name='mouth_downsampling1')
    mfc3 = tflearn.fully_connected(mfc3, 1024, activation='tanh', name='mouth_fc3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.conv_2d(mo_net, 1, 4, activation='tanh', name='mouth_downsampling2')
    mfc2 = tflearn.fully_connected(mfc2, 1024, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net = tflearn.conv_2d(mo_net, 1, 4, activation='tanh', name='mouth_downsampling3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')
    #mo_net=tf.concat([mo_net, mfc2, mfc3], 1, name='mouth_fc')#3072
    mo_net=tflearn.layers.merge([mo_net, mfc2, mfc3], 'concat', name='mouth_fc')

    #fc_net=tf.concat([e_net,mi_net,mo_net], 1, name='fusion_1')#9216
    fc_net=tflearn.layers.merge([e_net,mi_net,mo_net], 'concat', name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, classNo, activation='softmax', name='prob')
    return softmax

###using net 12
def FacePatches_NET_3Conv_3Inception_tflearn_12(eyep, middlep, mouthp, classNo=7):
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    efc3 = tflearn.conv_2d(e_net, 1, 2, activation='tanh', name='eye_downsampling1')
    efc3 = tflearn.fully_connected(efc3, 1024, activation='tanh', name='eye_fc3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.conv_2d(e_net, 1, 8, activation='tanh', name='eye_downsampling2')
    efc2 = tflearn.fully_connected(efc2, 1024, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net = tflearn.conv_2d(e_net, 1, 32, activation='tanh', name='eye_downsampling3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')
    #e_net=tf.concat([e_net, efc2, efc3], 1, name='eye_fc')#3072
    e_net=tflearn.layers.merge([e_net, efc2, efc3], 'concat', name='eye_fc')

    mi_net=tflearn.conv_2d(middlep, 8, 3, activation='relu',name='middle_conv1_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 8, 3, activation='relu',name='middle_conv1_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net,2,2,name='middle_pool1')
    mifc3 = tflearn.conv_2d(mi_net, 1, 2, activation='tanh', name='middle_downsampling1')
    mifc3 = tflearn.fully_connected(mifc3, 1024, activation='tanh', name='middle_fc3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 32, 3, activation='relu', name='middle_conv2_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool2')
    mifc2 = tflearn.conv_2d(mi_net, 1, 8, activation='tanh', name='middle_downsampling2')
    mifc2 = tflearn.fully_connected(mifc2, 1024, activation='tanh', name='middle_fc2')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_1_3x3')
    mi_net=tflearn.conv_2d(mi_net, 128, 3, activation='relu', name='middle_conv3_2_3x3')
    mi_net=tflearn.max_pool_2d(mi_net, 2, 2, name='middle_pool3')
    mi_net = tflearn.conv_2d(mi_net, 1, 32, activation='tanh', name='middle_downsampling3')
    mi_net=tflearn.fully_connected(mi_net, 1024, activation='tanh', name='middle_fc1')
    #mi_net=tf.concat([mi_net, mifc2, mifc3], 1, name='middle_fc')#3072
    mi_net=tflearn.layers.merge([mi_net, mifc2, mifc3], 'concat', name='middle_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mfc3 = tflearn.conv_2d(mo_net, 1, 2, activation='tanh', name='mouth_downsampling1')
    mfc3 = tflearn.fully_connected(mfc3, 1024, activation='tanh', name='mouth_fc3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.conv_2d(mo_net, 1, 8, activation='tanh', name='mouth_downsampling2')
    mfc2 = tflearn.fully_connected(mfc2, 1024, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net = tflearn.conv_2d(mo_net, 1, 32, activation='tanh', name='mouth_downsampling3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')
    #mo_net=tf.concat([mo_net, mfc2, mfc3], 1, name='mouth_fc')#3072
    mo_net=tflearn.layers.merge([mo_net, mfc2, mfc3], 'concat', name='mouth_fc')

    #fc_net=tf.concat([e_net,mi_net,mo_net], 1, name='fusion_1')#9216
    fc_net=tflearn.layers.merge([e_net,mi_net,mo_net], 'concat', name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, classNo, activation='softmax', name='prob')
    return softmax

###using net 24
def FacePatches_NET_3C_1I_2P(eyep, mouthp, classNo=7):
    ###using net 24
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')

    fc_net=tf.concat([e_net, mo_net], 1, name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, classNo, activation='softmax', name='prob')
    return softmax
###using net 25 
def FacePatches_NET_3C_2I_2P(eyep, mouthp, classNo=7):
    ###using net 25 
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')
    e_net=tf.concat([e_net, efc2], 1, name='eye_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')
    mo_net=tf.concat([mo_net, mfc2], 1, name='mouth_fc')

    fc_net=tf.concat([e_net, mo_net], 1, name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, classNo, activation='softmax', name='prob')
    return softmax
###using net 26
def FacePatches_NET_3C_3I_2P(eyep, mouthp, classNo=7):
    ###using net 26
    e_net=tflearn.conv_2d(eyep, 8, 3, activation='relu',name='eye_conv1_1_3x3')
    e_net=tflearn.conv_2d(e_net, 8, 3, activation='relu',name='eye_conv1_2_3x3')
    e_net=tflearn.max_pool_2d(e_net,2,2,name='eye_pool1')
    efc3 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_1_3x3')
    e_net=tflearn.conv_2d(e_net, 32, 3, activation='relu', name='eye_conv2_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool2')
    efc2 = tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc2')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_1_3x3')
    e_net=tflearn.conv_2d(e_net, 128, 3, activation='relu', name='eye_conv3_2_3x3')
    e_net=tflearn.max_pool_2d(e_net, 2, 2, name='eye_pool3')
    e_net=tflearn.fully_connected(e_net, 1024, activation='tanh', name='eye_fc1')
    e_net=tf.concat([e_net, efc2, efc3], 1, name='eye_fc')

    mo_net=tflearn.conv_2d(mouthp, 8, 3, activation='relu',name='mouth_conv1_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 8, 3, activation='relu',name='mouth_conv1_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net,2,2,name='mouth_pool1')
    mfc3 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 32, 3, activation='relu', name='mouth_conv2_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool2')
    mfc2 = tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc2')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_1_3x3')
    mo_net=tflearn.conv_2d(mo_net, 128, 3, activation='relu', name='mouth_conv3_2_3x3')
    mo_net=tflearn.max_pool_2d(mo_net, 2, 2, name='mouth_pool3')
    mo_net=tflearn.fully_connected(mo_net, 1024, activation='tanh', name='mouth_fc1')
    mo_net=tf.concat([mo_net, mfc2, mfc3], 1, name='mouth_fc')

    fc_net=tf.concat([e_net,mo_net], 1, name='fusion_1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc1')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop1')
    fc_net=tflearn.fully_connected(fc_net, 2048, activation='relu', name='fc2')
    fc_net=tflearn.dropout(fc_net, 0.8, name='drop2')
    softmax=tflearn.fully_connected(fc_net, classNo, activation='softmax', name='prob')
    return softmax