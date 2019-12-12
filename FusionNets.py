from NETWORK import Network

#N0
class VGG_Geo_NET(Network):
    def setup(self):
        (self.feed('image')#
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3')
             .max_pool(2, 2, 2, 2, name='pool5')
             .fc(4096, name='vggfc1'))

        (self.feed('geometry')#
             .hiddenLayer(4096, name = 'hidden1', relu=True)#
             .fc(2048, name='geofc1'))

        (self.feed('vggfc1', 'geofc1')
             .concat(1,name = 'fusion1')
             .fc(4096, name = 'fusion_fc1')
             .dropout(0.5, name = 'drop1')
             .fc(7, relu=False, name = 'fc7')
             .softmax(name='prob'))

class VGG_Geo_NET_N1(Network):
    def setup(self):
        (self.feed('image')#
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3')
             .max_pool(2, 2, 2, 2, name='pool5')
             .fc(4096, name='vggfc1'))

        (self.feed('geometry')#
             .hiddenLayer(4096, name = 'hidden1', relu=True)#
             .fc(2048, name='geofc1')
             .fc(7, name = 'geofc7'))

        (self.feed('vggfc1', 'geofc1')
             .concat(1,name = 'fusion1')
             .fc(2048, name = 'fusion_fc1')
             .dropout(0.5, name = 'drop1')
             .fc(7, relu=False, name = 'fc7')
             .softmax(name='prob'))