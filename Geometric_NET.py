from NETWORK import Network
'''
for JAFFE incomplete tests, 
Geometric_NET_1h > Geometric_NET_2h1I > Geometric_NET_2c2l > 
Geometric_NET_3h1I > Geometric_NET_2c2lcc1l1 > Geometric_NET_2c2lcc1
'''
#N0
class Geometric_NET_2c2l(Network):
    def setup(self):
        (self.feed('data')#92
             .conv1D(1, 32, 1, name = 'conv1')#92*32
             .conv1D(2, 64, 1, name = 'conv2'))#92*64

        (self.feed('conv1', 'conv2')
             .concat(2, name = 'inception')#92*96 = 17664
             .fc(2048, name='gefc1')
             .dropout(0.5, name='gedrop1')
             .fc(1024, name='gefc2')
             .dropout(0.5, name = 'gedrop2')
             .fc(7, relu = False, name = 'gefc7')
             .softmax(name='geprob'))

#N1
class Geometric_NET_2c2lcc1(Network):
    def setup(self):
        (self.feed('data')#92
             .conv1D(1, 32, 1, name = 'conv1')#92*32
             .conv1D(2, 64, 1, name = 'conv2'))#92*64

        (self.feed('conv1', 'conv2')
             .concat(2, name = 'inception')#92*192
             .conv1D(1, 256, 1, name = 'inception_conv1')#92*256 = 23552
             .fc(2048, name='gefc1')
             .dropout(0.5, name='gedrop1')
             .fc(1024, name='gefc2')
             .dropout(0.5, name = 'gedrop2')
             .fc(7, relu = False, name = 'gefc7')
             .softmax(name='geprob'))

#N2
class Geometric_NET_2c2lcc1l1(Network):
    def setup(self):
        (self.feed('data')#92
             .conv1D(1, 32, 1, name = 'conv1')#92*32
             .conv1D(1, 96, 1, name = 'conv2'))#92*64

        (self.feed('conv1', 'conv2')
             .concat(2, name = 'inception')#92*192
             .conv1D(1, 256, 1, name = 'inception_conv1')#92*256 
             .fc(2048, name='gefc1')
             .dropout(0.5, name='gedrop1')
             .fc(1024, name='gefc2')
             .dropout(0.5, name = 'gedrop2')
             .fc(7, relu = False, name = 'gefc7')
             .softmax(name='geprob'))

#N3
class Geometric_NET_1h(Network):
    def setup(self):
        (self.feed('data')#92
             .hiddenLayer(4096, name = 'hidden1')#4096
             .fc(1024, name='gefc2')
             .dropout(0.5, name = 'gedrop1')
             .fc(7, relu = False, name = 'gefc7')
             .softmax(name='geprob'))

#N4
class Geometric_NET_2h1I(Network):
    def setup(self):
        (self.feed('data')#122
             .hiddenLayer(4096, name = 'hidden1', relu=True)#4096
             .fc(2048, name='gefc1')
             .dropout(0.5, name='gedrop1')
             .fc(1024, name='gefc2')
             .dropout(0.5, name = 'gedrop2')
             .fc(7, relu = False, name = 'gefc7')
             .softmax(name='geprob'))

#N5
class Geometric_NET_3h1I(Network):
    def setup(self):
        (self.feed('data')#92
             .hiddenLayer(4096, name = 'hidden1')#4096
             .hiddenLayer(1024, name = 'hidden2'))#1024

        (self.feed('hidden1', 'hidden2')
             .concat(1, name = 'inception')#5120
             .hiddenLayer(6144, name='hidden3')
             .fc(2048, name='gefc1')
             .dropout(0.5, name='gedrop1')
             .fc(1024, name='gefc2')
             .dropout(0.5, name = 'gedrop2')
             .fc(7, relu = False, name = 'gefc7')
             .softmax(name='geprob'))

#N6
class Geometric_NET_h1I(Network):
    def setup(self):
        (self.feed('data')#122
             .hiddenLayer(512, name = 'hidden1', relu=True)#4096
             .fc(256, name='gefc1')
             .dropout(0.5, name='gedrop1')
             .fc(128, name='gefc2')
             .dropout(0.5, name = 'gedrop2')
             .fc(7, relu = False, name = 'gefc7')
             .softmax(name='geprob'))