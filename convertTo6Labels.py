import os, pickle, time, sys, traceback
import DataSetPrepare as DSP
from DataSetPrepare import Dataset_Dictionary
datalist=[]

c=len(sys.argv)
if c>1:
    for i in range(1, c):
        print(sys.argv[i])
        if int(sys.argv[i])>60000:
            continue
        r=Dataset_Dictionary.get(int(sys.argv[i]), None)
        #print(r)
        if r is None:
            print('Unexpected Dataset ID encount.')
        else:
            datalist.append(r)
    t1=time.time()
    DSP.loadandConvertTo6LabelsWithoutNatual(datalist)
    t2=time.time()
    print('Time consumed: %fs'%(t2-t1))
else:
    print('Usage: python convertTo6Labels.py datasetID1 datasetID2 ...')
    