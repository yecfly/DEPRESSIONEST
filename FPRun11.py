from Facepatchindependenttrain import runPatch
import sys

if len(sys.argv)==6:
    runPatch(GPU_Device_ID=1, FacePatchID=int(sys.argv[1]), 
                    trainpklID=int(sys.argv[2]), testpklID=int(sys.argv[3]),
                    NetworkType=int(sys.argv[4]), 
                    runs=int(sys.argv[5]))
else:
    print("argument errors, try\npython runfile.py <FacePatchID> <trainpklID> <testpklID> <NetworkType> <runs>")
