mkdir ../weights/20181211
mkdir ../logs/20181211

#SLO lr=0.0001
python kitti.py --func=train --net=slo --merge=add --check=true --augm=true --load=true --epochs=100 --lr=0.0001 --folder=20181211 > ../logs/20181211/20181211_slo_add_0001.txt

python kitti.py --func=train --net=slo --merge=avg --check=true --augm=true --load=true --epochs=100 --lr=0.0001 --folder=20181211 > ../logs/20181211/20181211_slo_avg_0001.txt

python kitti.py --func=train --net=slo --merge=max --check=true --augm=true --load=true --epochs=100 --lr=0.0001 --folder=20181211 > ../logs/20181211/20181211_slo_max_0001.txt


#ALO lr=0.0001
python kitti.py --func=train --net=alo --merge=add --check=true --augm=true --load=true --epochs=100 --lr=0.0001 --folder=20181211 > ../logs/20181211/20181211_alo_add_0001.txt

python kitti.py --func=train --net=alo --merge=avg --check=true --augm=true --load=true --epochs=100 --lr=0.0001 --folder=20181211 > ../logs/20181211/20181211_alo_avg_0001.txt

python kitti.py --func=train --net=alo --merge=max --check=true --augm=true --load=true --epochs=100 --lr=0.0001 --folder=20181211 > ../logs/20181211/20181211_alo_max_0001.txt
