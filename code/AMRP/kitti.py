
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
parser.add_argument("--epochs", nargs='?', type=int)
parser.add_argument("--lr", nargs='?', type=float)
parser.add_argument("--morf", nargs='?', type=str2bool)
#parser.add_argument("--balanced", nargs='?', type=str2bool)
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

#epochs parameter
nb_epochs = 100
if(args.epochs != None and args.epochs > 0):
    nb_epochs = args.epochs

#learning rate parameter
learn_rate = 0.001
if(args.lr != None and args.lr > 0):
    learn_rate = args.lr

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
print('Epochs: ', nb_epochs)
print('Learning rate: ', learn_rate)
print('Math morfology: ', args.morf)
#print('Balanced: ', args.balanced)
print('-----END PARAMETERS-----')

#############
#net = {full, hed}
if(args.net != None):
    import model_kitti as mk
    if(args.net == 'hed'):
        model = mk.model_hed(merge_name=args.merge, vote_value=args.vote, morf=False) #, morf=args.morf)
    elif(args.net == 'full'):
        model = mk.model_full()
    else:
        printWrongUsageAndQuit()

#func = {train, test, npy}
if(args.func == 'train'):
    import train_kitti as trk
    trk.train(model=model, net=args.net, merge=args.merge, vote=vote_value, check=args.check, augm=args.augm
        , load=args.load, nb_epoch=nb_epochs, learn_rate=learn_rate) #, balanced=args.balanced)

elif(args.func == 'npy'):
    import npy_kitti as nk
    nk.npy(set_name=args.set, augm=args.augm)

elif(args.func == 'test'):
    import test_kitti as tek
    tek.test(model=model, net=args.net, merge_name=args.merge, set_name=args.set, mark=args.mark, morf=args.morf)

else:
    printWrongUsageAndQuit()
