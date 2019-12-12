from NETWORK import Network
import tflearn as tfl
import tensorflow as tf
'''##N0
#class VGG_NET_20l_512o(Network):
#    def setup(self):
#        (self.feed('data')#128*128*1
#             .conv(3, 3, 16, 1, 1, name='conv1_1')#128*128*16
#             .conv(3, 3, 16, 1, 1, name='conv1_2')#128*128*16
#             .max_pool(2, 2, 2, 2, name='pool1')#64*64*16
#             .conv(3, 3, 32, 1, 1, name='conv2_1')#64*64*32
#             .conv(3, 3, 32, 1, 1, name='conv2_2')#64*64*32
#             .max_pool(2, 2, 2, 2, name='pool2')#32*32*32
#             .conv(3, 3, 64, 1, 1, name='conv3_1')#32*32*64
#             .conv(3, 3, 64, 1, 1, name='conv3_2')#32*32*64
#             .conv(3, 3, 64, 1, 1, name='conv3_3')#32*32*64
#             .max_pool(2, 2, 2, 2, name='pool3')#16*16*64
#             .conv(3, 3, 128, 1, 1, name='conv4_1')#16*16*128
#             .conv(3, 3, 128, 1, 1, name='conv4_2')#16*16*128
#             .conv(3, 3, 128, 1, 1, name='conv4_3')#16*16*128
#             .max_pool(2, 2, 2, 2, name='pool4')#8*8*128
#             .conv(3, 3, 128, 1, 1, name='conv5_1')#8*8*128
#             .conv(3, 3, 128, 1, 1, name='conv5_2')#8*8*128
#             .conv(3, 3, 128, 1, 1, name='conv5_3')#8*8*128
#             .max_pool(2, 2, 2, 2, name='pool5')#4*4*128
#             .fc(1024, name='fc1')
#             .dropout(0.5, name='drop1')
#             .fc(512, name='fc2')
#             .dropout(0.5, name = 'drop2')
#             .fc(7, relu = False, name = 'fc7')
#             .softmax(name='prob'))
#    #def setup(self):
#        #(self.feed('data')#224*224
#             #.conv(3, 3, 64, 1, 1, name='conv1_1')#224*224*64
#             #.conv(3, 3, 64, 1, 1, name='conv1_2')#224*224*64
#             #.max_pool(2, 2, 2, 2, name='pool1')#112*112*64
#             #.conv(3, 3, 128, 1, 1, name='conv2_1')#112*112*128
#             #.conv(3, 3, 128, 1, 1, name='conv2_2')#112*112*128
#             #.max_pool(2, 2, 2, 2, name='pool2')#56*56*128
#             #.conv(3, 3, 256, 1, 1, name='conv3_1')#56*56*256
#             #.conv(3, 3, 256, 1, 1, name='conv3_2')#56*56*256
#             #.conv(3, 3, 256, 1, 1, name='conv3_3')#56*56*256
#             #.max_pool(2, 2, 2, 2, name='pool3')#28*28*256
#             #.conv(3, 3, 512, 1, 1, name='conv4_1')#28*28*512
#             #.conv(3, 3, 512, 1, 1, name='conv4_2')#28*28*512
#             #.conv(3, 3, 512, 1, 1, name='conv4_3')#28*28*512
#             #.max_pool(2, 2, 2, 2, name='pool4')#14*14*512
#             #.conv(3, 3, 512, 1, 1, name='conv5_1')#14*14*512
#             #.conv(3, 3, 512, 1, 1, name='conv5_2')#14*14*512
#             #.conv(3, 3, 512, 1, 1, name='conv5_3')#14*14*512
#             #.max_pool(2, 2, 2, 2, name='pool5')#7*7*512
#        #     .fc(4096, name='fc6')
#        #     .dropout(0.5, name='drop6')
#        #     .fc(4096, name='fc7')
#        #     .dropout(0.5, name='drop7')
#        #     .fc(2622, relu=False, name='fc8')
#        #     .softmax(name='prob'))

##N1
#class VGG_NET_16l_72o(Network):
#    def setup(self):
#        (self.feed('data')#128*128
#             .conv(3, 3, 8, 1, 1, name='conv1_1')#128*128*8
#             .conv(3, 3, 8, 1, 1, name='conv1_2')#128*128*8
#             .max_pool(2, 2, 2, 2, name='pool1')#64*64*8
#             .conv(3, 3, 16, 1, 1, name='conv2_1')#64*64*16
#             .conv(3, 3, 16, 1, 1, name='conv2_2')#64*64*16
#             .max_pool(2, 2, 2, 2, name='pool2')#32*32*16
#             .conv(3, 3, 32, 1, 1, name='conv3_1')#32*32*32
#             .conv(3, 3, 32, 1, 1, name='conv3_2')#32*32*32
#             .conv(3, 3, 32, 1, 1, name='conv3_3')#32*32*32
#             .max_pool(2, 2, 2, 2, name='pool3')#16*16*32
#             .conv(3, 3, 64, 1, 1, name='conv4_1')#16*16*64
#             .conv(3, 3, 64, 1, 1, name='conv4_2')#16*16*64
#             .conv(3, 3, 64, 1, 1, name='conv4_3')#16*16*64
#             .max_pool(2, 2, 2, 2, name='pool4')#8*8*64
#             .fc(288, name='fc1')
#             .dropout(0.5, name='drop1')
#             .fc(72, name='fc2')
#             .dropout(0.5, name = 'drop2')
#             .fc(7, relu = False, name = 'fc7')
#             .softmax(name='prob'))
'''
#N2
class VGG_NET_16l_128o(Network):
    def setup(self):
        (self.feed('data')#128*128
             .conv(3, 3, 8, 1, 1, name='conv1_1')#128*128*8
             .conv(3, 3, 8, 1, 1, name='conv1_2')#128*128*8
             .max_pool(2, 2, 2, 2, name='pool1')#64*64*8
             .conv(3, 3, 16, 1, 1, name='conv2_1')#64*64*16
             .conv(3, 3, 16, 1, 1, name='conv2_2')#64*64*16
             .max_pool(2, 2, 2, 2, name='pool2')#32*32*16
             .conv(3, 3, 32, 1, 1, name='conv3_1')#32*32*32
             .conv(3, 3, 32, 1, 1, name='conv3_2')#32*32*32
             .conv(3, 3, 32, 1, 1, name='conv3_3')#32*32*32
             .max_pool(2, 2, 2, 2, name='pool3')#16*16*32
             .conv(3, 3, 64, 1, 1, name='conv4_1')#16*16*64
             .conv(3, 3, 64, 1, 1, name='conv4_2')#16*16*64
             .conv(3, 3, 64, 1, 1, name='conv4_3')#16*16*64
             .max_pool(2, 2, 2, 2, name='pool4')#8*8*64
             .fc(512, name='fc1')
             .dropout(0.5, name='drop1')
             .fc(128, name='fc2')
             .dropout(0.5, name = 'drop2')
             .fc(7, relu = False, name = 'fc7')
             .softmax(name='prob'))

##N3
#class VGG_NET_20l_128o(Network):
#    def setup(self):
#        (self.feed('data')#128*128
#             .conv(3, 3, 8, 1, 1, name='conv1_1')#128*128*8
#             .conv(3, 3, 8, 1, 1, name='conv1_2')#128*128*8
#             .max_pool(2, 2, 2, 2, name='pool1')#64*64*8
#             .conv(3, 3, 16, 1, 1, name='conv2_1')#64*64*16
#             .conv(3, 3, 16, 1, 1, name='conv2_2')#64*64*16
#             .max_pool(2, 2, 2, 2, name='pool2')#32*32*16
#             .conv(3, 3, 32, 1, 1, name='conv3_1')#32*32*32
#             .conv(3, 3, 32, 1, 1, name='conv3_2')#32*32*32
#             .conv(3, 3, 32, 1, 1, name='conv3_3')#32*32*32
#             .max_pool(2, 2, 2, 2, name='pool3')#16*16*32
#             .conv(3, 3, 64, 1, 1, name='conv4_1')#16*16*64
#             .conv(3, 3, 64, 1, 1, name='conv4_2')#16*16*64
#             .conv(3, 3, 64, 1, 1, name='conv4_3')#16*16*64
#             .max_pool(2, 2, 2, 2, name='pool4')#8*8*64
#             .conv(3, 3, 64, 1, 1, name='conv5_1')#8*8*64
#             .conv(3, 3, 64, 1, 1, name='conv5_2')#8*8*64
#             .conv(3, 3, 64, 1, 1, name='conv5_3')#8*8*64
#             .max_pool(2, 2, 2, 2, name='pool5')#4*4*64
#             .fc(512, name='fc1')
#             .dropout(0.5, name='drop1')
#             .fc(128, name='fc2')
#             .dropout(0.5, name = 'drop2')
#             .fc(7, relu = False, name = 'fc7')
#             .softmax(name='prob'))

#N4
class VGG_NET_o(Network):
    def setup(self):
        (self.feed('data')#224*224
             .conv(3, 3, 64, 1, 1, name='conv1_1')#224*224*64
             .conv(3, 3, 64, 1, 1, name='conv1_2')#224*224*64
             .max_pool(2, 2, 2, 2, name='pool1')#112*112*64
             .conv(3, 3, 128, 1, 1, name='conv2_1')#112*112*128
             .conv(3, 3, 128, 1, 1, name='conv2_2')#112*112*128
             .max_pool(2, 2, 2, 2, name='pool2')#56*56*128
             .conv(3, 3, 256, 1, 1, name='conv3_1')#56*56*256
             .conv(3, 3, 256, 1, 1, name='conv3_2')#56*56*256
             .conv(3, 3, 256, 1, 1, name='conv3_3')#56*56*256
             .max_pool(2, 2, 2, 2, name='pool3')#28*28*256
             .conv(3, 3, 512, 1, 1, name='conv4_1')#28*28*512
             .conv(3, 3, 512, 1, 1, name='conv4_2')#28*28*512
             .conv(3, 3, 512, 1, 1, name='conv4_3')#28*28*512
             .max_pool(2, 2, 2, 2, name='pool4')#14*14*512
             .conv(3, 3, 512, 1, 1, name='conv5_1')#14*14*512
             .conv(3, 3, 512, 1, 1, name='conv5_2')#14*14*512
             .conv(3, 3, 512, 1, 1, name='conv5_3')#14*14*512
             .max_pool(2, 2, 2, 2, name='pool5')#7*7*512
             .fc(4096, name='fc1')
             .dropout(0.5, name='drop6')
             .fc(4096, name='fc2')
             .dropout(0.5, name = 'drop2')
             .fc(7, relu = False, name = 'fc7')
             .softmax(name='prob'))

#N5
class PRANDRE(Network):
    def setup(self):
        (self.feed('data')#
             .conv(5, 5, 32, 1, 1, name='conv1')#
             .max_pool(2, 2, 2, 2, name='pool1')#
             .conv(7, 7, 64, 1, 1, name='conv2')#
             .max_pool(2, 2, 2, 2, name='pool2')#
             .fc(256, name='fc6')
             .dropout(0.5, name = 'drop2')
             .fc(7, relu = False, name = 'fc7')
             .softmax(name='prob'))

#N6
class PRANDRE1(Network):
    def setup(self):
        (self.feed('data')#
             .conv(5, 5, 64, 1, 1, name='conv1')#
             .max_pool(2, 2, 2, 2, name='pool1')#
             .conv(7, 7, 128, 1, 1, name='conv2')#
             .max_pool(2, 2, 2, 2, name='pool2')#
             .fc(512, name='fc6')
             .dropout(0.5, name = 'drop2')
             .fc(7, relu = False, name = 'fc7')
             .softmax(name='prob'))

#N7
class PRANDRE2(Network):
    def setup(self):
        (self.feed('data')#
             .conv(5, 5, 128, 1, 1, name='conv1')#
             .max_pool(2, 2, 2, 2, name='pool1')#
             .conv(7, 7, 256, 1, 1, name='conv2')#
             .max_pool(2, 2, 2, 2, name='pool2')#
             .fc(2048, name='fc6')
             .dropout(0.5, name = 'drop2')
             .fc(7, relu = False, name = 'fc7')
             .softmax(name='prob'))

#N8
class VGG_NET_Inception1(Network):
    def setup(self):
        (self.feed('data')#224*224
             .conv(3, 3, 64, 1, 1, name='conv1_1')#224*224*64
             .conv(3, 3, 64, 1, 1, name='conv1_2')#224*224*64
             .max_pool(2, 2, 2, 2, name='pool1')#112*112*64
             .conv(3, 3, 128, 1, 1, name='conv2_1')#112*112*128
             .conv(3, 3, 128, 1, 1, name='conv2_2')#112*112*128
             .max_pool(2, 2, 2, 2, name='pool2')#56*56*128
             .conv(3, 3, 256, 1, 1, name='conv3_1')#56*56*256
             .conv(3, 3, 256, 1, 1, name='conv3_2')#56*56*256
             .conv(3, 3, 256, 1, 1, name='conv3_3')#56*56*256
             .max_pool(2, 2, 2, 2, name='pool3')#28*28*256
             .conv(3, 3, 512, 1, 1, name='conv4_1')#28*28*512
             .conv(3, 3, 512, 1, 1, name='conv4_2')#28*28*512
             .conv(3, 3, 512, 1, 1, name='conv4_3')#28*28*512
             .max_pool(2, 2, 2, 2, name='pool4')#14*14*512
             .conv(3, 3, 512, 1, 1, name='conv5_1')#14*14*512
             .conv(3, 3, 512, 1, 1, name='conv5_2')#14*14*512
             .conv(3, 3, 512, 1, 1, name='conv5_3')#14*14*512
             .max_pool(2, 2, 2, 2, name='pool5')#7*7*512
             .fc(4096, name='fc1')
             .dropout(0.5, name='drop6')
             .fc(4096, name='fc2'))

        (self.feed('pool3')
             .fc(512, name = 'pool2_fc'))

        (self.feed('fc2','pool2_fc')
             .concat(1,name = 'fc1_pool2fc')
             .dropout(0.5, name = 'drop2')
             .fc(7, relu = False, name = 'fc7')
             .softmax(name='prob'))

#N9
class VGG_NET_Inception2(Network):
    def setup(self):
        (self.feed('data')#224*224
             .conv(3, 3, 64, 1, 1, name='conv1_1')#224*224*64
             .conv(3, 3, 64, 1, 1, name='conv1_2')#224*224*64
             .max_pool(2, 2, 2, 2, name='pool1')#112*112*64
             .conv(3, 3, 128, 1, 1, name='conv2_1')#112*112*128
             .conv(3, 3, 128, 1, 1, name='conv2_2')#112*112*128
             .max_pool(2, 2, 2, 2, name='pool2')#56*56*128
             .conv(3, 3, 256, 1, 1, name='conv3_1')#56*56*256
             .conv(3, 3, 256, 1, 1, name='conv3_2')#56*56*256
             .conv(3, 3, 256, 1, 1, name='conv3_3')#56*56*256
             .max_pool(2, 2, 2, 2, name='pool3')#28*28*256
             .conv(3, 3, 512, 1, 1, name='conv4_1')#28*28*512
             .conv(3, 3, 512, 1, 1, name='conv4_2')#28*28*512
             .conv(3, 3, 512, 1, 1, name='conv4_3')#28*28*512
             .max_pool(2, 2, 2, 2, name='pool4')#14*14*512
             .conv(3, 3, 512, 1, 1, name='conv5_1')#14*14*512
             .conv(3, 3, 512, 1, 1, name='conv5_2')#14*14*512
             .conv(3, 3, 512, 1, 1, name='conv5_3')#14*14*512
             .max_pool(2, 2, 2, 2, name='pool5')#7*7*512
             .fc(4096, name='fc1')
             .dropout(0.5, name='drop6')
             .fc(4096, name='fc2'))

        (self.feed('pool4')
             .fc(512, name = 'pool2_fc'))

        (self.feed('fc2','pool2_fc')
             .concat(1,name = 'fc1_pool2fc')
             .dropout(0.5, name = 'drop2')
             .fc(7, relu = False, name = 'fc7')
             .softmax(name='prob'))

#10
def VGG_NET_O_tfl(img):
    vgg_net = tfl.conv_2d(img, 64, 3, activation='relu', name='conv1_1')
    vgg_net = tfl.conv_2d(vgg_net, 64, 3, activation='relu', name='conv1_2')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool1')
    vgg_net = tfl.conv_2d(vgg_net, 128, 3, activation='relu', name='conv2_1')
    vgg_net = tfl.conv_2d(vgg_net, 128, 3, activation='relu', name='conv2_2')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool2')
    vgg_net = tfl.conv_2d(vgg_net, 256, 3, activation='relu', name='conv3_1')
    vgg_net = tfl.conv_2d(vgg_net, 256, 3, activation='relu', name='conv3_2')
    vgg_net = tfl.conv_2d(vgg_net, 256, 3, activation='relu', name='conv3_3')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool3')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='relu', name='conv4_1')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='relu', name='conv4_2')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='relu', name='conv4_3')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool4')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='relu', name='conv5_1')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='relu', name='conv5_2')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='relu', name='conv5_3')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool5')
    vgg_net = tfl.fully_connected(vgg_net, 4096, activation='tanh', name='fc1')
    vgg_net = tfl.dropout(vgg_net, 0.8, name='drop1')
    vgg_net = tfl.fully_connected(vgg_net, 4096, activation='tanh', name='fc2')
    vgg_net = tfl.dropout(vgg_net, 0.8, name='drop2')
    softmax = tfl.fully_connected(vgg_net, 7, activation='softmax', name='prob')

    return softmax

#11
def VGG_NET_I5(img):
    vgg_net = tfl.conv_2d(img, 64, 3, activation='relu', name='conv1_1')
    vgg_net = tfl.conv_2d(vgg_net, 64, 3, activation='relu', name='conv1_2')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool1')
    #fcpool1 = tfl.fully_connected(vgg_net, 1024, activation='tanh', name='fcpool1')
    vgg_net = tfl.conv_2d(vgg_net, 128, 3, activation='relu', name='conv2_1')
    vgg_net = tfl.conv_2d(vgg_net, 128, 3, activation='relu', name='conv2_2')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool2')
    #fcpool2 = tfl.fully_connected(vgg_net, 1024, activation='tanh', name='fcpool2')
    vgg_net = tfl.conv_2d(vgg_net, 256, 3, activation='relu', name='conv3_1')
    vgg_net = tfl.conv_2d(vgg_net, 256, 3, activation='relu', name='conv3_2')
    vgg_net = tfl.conv_2d(vgg_net, 256, 3, activation='relu', name='conv3_3')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool3')
    #fcpool3 = tfl.fully_connected(vgg_net, 1024, activation='tanh', name='fcpool3')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='relu', name='conv4_1')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='relu', name='conv4_2')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='relu', name='conv4_3')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool4')
    fcpool4 = tfl.fully_connected(vgg_net, 1024, activation='tanh', name='fcpool4')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='relu', name='conv5_1')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='relu', name='conv5_2')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='relu', name='conv5_3')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool5')
    fcpool5 = tfl.fully_connected(vgg_net, 4096, activation='tanh', name='fcpool5')

    #fc_net=tf.concat([fcpool1,fcpool2, fcpool3, fcpool4, fcpool5], 1, name='fusion_1')
    fc_net=tf.concat([fcpool4, fcpool5], 1, name='fusion_1')
    
    fc_net = tfl.fully_connected(fc_net, 2048, activation='tanh', name='fc1')
    fc_net = tfl.dropout(fc_net, 0.8, name='drop1')
    fc_net = tfl.fully_connected(fc_net, 2048, activation='tanh', name='fc2')
    fc_net = tfl.dropout(fc_net, 0.8, name='drop2')
    softmax = tfl.fully_connected(fc_net, 7, activation='softmax', name='prob')

    return softmax

#12
def VGG_NET_I5_ELU(img):
    vgg_net = tfl.conv_2d(img, 64, 3, activation='ELU', name='conv1_1')
    vgg_net = tfl.conv_2d(vgg_net, 64, 3, activation='ELU', name='conv1_2')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool1')
    #fcpool1 = tfl.fully_connected(vgg_net, 1024, activation='tanh', name='fcpool1')
    vgg_net = tfl.conv_2d(vgg_net, 128, 3, activation='ELU', name='conv2_1')
    vgg_net = tfl.conv_2d(vgg_net, 128, 3, activation='ELU', name='conv2_2')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool2')
    #fcpool2 = tfl.fully_connected(vgg_net, 1024, activation='tanh', name='fcpool2')
    vgg_net = tfl.conv_2d(vgg_net, 256, 3, activation='ELU', name='conv3_1')
    vgg_net = tfl.conv_2d(vgg_net, 256, 3, activation='ELU', name='conv3_2')
    vgg_net = tfl.conv_2d(vgg_net, 256, 3, activation='ELU', name='conv3_3')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool3')
    #fcpool3 = tfl.fully_connected(vgg_net, 1024, activation='tanh', name='fcpool3')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='ELU', name='conv4_1')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='ELU', name='conv4_2')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='ELU', name='conv4_3')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool4')
    fcpool4 = tfl.fully_connected(vgg_net, 1024, activation='ELU', name='fcpool4')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='ELU', name='conv5_1')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='ELU', name='conv5_2')
    vgg_net = tfl.conv_2d(vgg_net, 512, 3, activation='ELU', name='conv5_3')
    vgg_net = tfl.max_pool_2d(vgg_net, 2, 2, name='pool5')
    fcpool5 = tfl.fully_connected(vgg_net, 4096, activation='ELU', name='fcpool5')

    #fc_net=tf.concat([fcpool1,fcpool2, fcpool3, fcpool4, fcpool5], 1, name='fusion_1')
    fc_net=tf.concat([fcpool4, fcpool5], 1, name='fusion_1')
    
    fc_net = tfl.fully_connected(fc_net, 2048, activation='ELU', name='fc1')
    fc_net = tfl.dropout(fc_net, 0.8, name='drop1')
    fc_net = tfl.fully_connected(fc_net, 2048, activation='ELU', name='fc2')
    fc_net = tfl.dropout(fc_net, 0.8, name='drop2')
    softmax = tfl.fully_connected(fc_net, 7, activation='softmax', name='prob')

    return softmax