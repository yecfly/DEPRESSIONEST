import glob
import os
if __name__ == '__main__':
    filew=open('./Datasets/dataset.txt','w')
    filew.write("Dataset_Dictionary={")
    for i, v in enumerate( glob.glob(os.path.join('./Datasets','*.pkl'))):
        tname=v.split('_')[0]
        print(tname)
        DID=int(tname[12:])
        print(DID, v)
        if i==0:
            filew.write("%d:'%s'"%(DID, v))
        else:
            filew.write("\n\t,%d:'%s'"%(DID, v))
    filew.write("\n}\n")
    filew.close()