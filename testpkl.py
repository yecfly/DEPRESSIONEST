import part2predict as P
import sys

if len(sys.argv)==5:
    P.TestPreprocessPKL(GPU_Device_ID=1, Dataset=int(sys.argv[1]), 
                        Module=int(sys.argv[2]), NetworkType=int(sys.argv[3]), 
                        TrainDataset=int(sys.argv[4]))
elif len(sys.argv)==6:
    P.TestPreprocessPKL(GPU_Device_ID=1, Dataset=int(sys.argv[1]), 
                        Module=int(sys.argv[2]), NetworkType=int(sys.argv[3]), 
                        TrainDataset=int(sys.argv[4]), saveSamples=bool(int(sys.argv[5])))
elif len(sys.argv)==7:
    P.TestPreprocessPKL(GPU_Device_ID=1, Dataset=int(sys.argv[1]), 
                        Module=int(sys.argv[2]), NetworkType=int(sys.argv[3]), 
                        TrainDataset=int(sys.argv[4]), saveSamples=bool(int(sys.argv[5])),
                        ModelName=str(sys.argv[6]))
else:
    print('Usage:\npython testpkl.py {Dataset} {Module} {NetworkType} {TrainDataset}')