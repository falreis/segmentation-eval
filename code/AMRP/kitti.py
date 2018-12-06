
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True

def printWrongUsageAndQuit():
    print('WRONG USAGE: Please read the README file.')
    quit()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--func", type=str)
parser.add_argument("--net", nargs='?', type = str)
parser.add_argument("--set", nargs='?', type=str)
parser.add_argument("--merge", nargs='?', type = str)
parser.add_argument("--vote", nargs='?', type= int)
parser.add_argument("--check", nargs='?', type= str2bool)
parser.add_argument("--augm", nargs='?', type=str2bool)
parser.add_argument("--load", nargs='?', type=str2bool)
parser.add_argument("--mark", nargs='?', type=str2bool)
args = parser.parse_args()

#func and net parameters
if(args.func == None):
    printWrongUsageAndQuit()

#net parameter
if(args.func in ('run', 'eval') and args.net == None):
    printWrongUsageAndQuit()

#merge parameter
if(args.net == 'hed' and args.merge == None):
    printWrongUsageAndQuit()

#vote parameter
vote_value = 0
if(args.vote):
    vote_value = args.vote
else:
    if(args.merge == "maj"):
        print('Maj operation must contain "--vote" parameter!')
        quit()

#print parameters
print('-----BEGIN PARAMETERS-----')
print('Functionality: ', args.func)
print('Neural net: ', args.net)
print('Set value: ', args.set)
print('Merge method: ', args.merge) 
print('Vote value: ', args.vote)
print('Use checkpoint: ', args.check)
print('Use data augm: ', args.augm)
print('Load vgg weights: ', args.load)
print('Mark road: ', args.mark)
print('-----END PARAMETERS-----')

##
#code
if(args.func == 'train'):
    import model_kitti as mk
    import train_kitti as trk

    if(args.net == 'hed'):
        trk.model_hed(merge_name=args.merge, vote_value=args.vote)
    elif(args.net == 'full'):
        trk.model_full()
    else:
        printWrongUsageAndQuit()

    rk.run(net=args.net, merge=args.merge, vote=vote_value, check=args.check, augm=args.augm, load=args.load)

elif(args.func == 'npy'):
    import npy_kitti as nk
    nk.npy(set_name=args.set, augm=args.augm)

elif(args.func == 'test'):
    import test_kitti as tek
    tek.eval(net=args.net, merge_name=args.merge, set_name=args.set, mark=args.mark)

else:
    printWrongUsageAndQuit()
