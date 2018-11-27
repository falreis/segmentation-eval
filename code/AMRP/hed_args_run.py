import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True

parser = argparse.ArgumentParser()
parser.add_argument("--net", type = str)
parser.add_argument("--merge", type = str)
parser.add_argument("--vote", nargs='?', type= int)
parser.add_argument("--check", nargs='?', type= str2bool)
parser.add_argument("--out", nargs='?', type=int)
args = parser.parse_args()
net_parse = args.net
merge_name = args.merge
check_value = args.check

if(args.vote):
    vote_value = args.vote
    print('Vote value: ', vote_value)
else:
    if(merge_name == "maj"):
        print('Maj operation must contain "--vote" parameter!')
        quit()
#parameters
if(args.out):
    out_value = args.out
    print('Out value: ', out_value)
else:
    out_value = 0

#se merge não for definido
if(merge_name == None):
    print('Usage >> "python run-kitti.py --net={hed,rcf} --merge={sum,avg,max,maj} --vote{0-5}"')
    quit()

print('Neural net: ', net_parse)
print('Merge method: ', merge_name)