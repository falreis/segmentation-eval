import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--merge", type = str)
parser.add_argument("--vote", nargs='?', type= int)
parser.add_argument("--out", nargs='?', type=int)
args = parser.parse_args()
merge_name = args.merge
print('Merge method: ', merge_name)

#parameters
if(args.out):
    out_value = args.out
    print('Out value: ', out_value)
else:
    out_value = 0

if(args.vote):
    vote_value = args.vote
    print('Vote value: ', vote_value)
else:
    if(merge_name == "maj"):
        print('Maj operation must contain "--vote" parameter!')
        quit()

#se merge não for definido
if(merge_name == None):
    print('Usage >> "python model-kitti_hed.py --merge={sum,avg,max,maj} --vote?=2 --out?=0"')
    quit()
