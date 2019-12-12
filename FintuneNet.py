from NETWORK import Network

#Nxx0
class FTN0(Network):
    def setup(self):
        (self.feed('appfc', 'geofc')
             .concat(1,name = 'fin_fusion1')
             .fc(4096, name = 'fin_fusion_fc1')
             .dropout(0.5, name = 'fin_drop1')
             .fc(7, relu=False, name = 'fin_fc7')
             .softmax(name='fin_prob'))

#Nxx1
class FTN1(Network):
    def setup(self):
        (self.feed('appfc', 'geofc')
             .concat(1,name = 'fin_fusion1')
             .fc(4096, name = 'fin_fusion_fc1')
             .dropout(0.5, name = 'fin_drop1')
             .fc(4096, name = 'fin_fusion_fc2')
             .dropout(0.5, name = 'fin_drop2')
             .fc(7, relu=False, name = 'fin_fc7')
             .softmax(name='fin_prob'))