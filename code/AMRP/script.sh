mkdir ../weights/20181211
mkdir ../logs/20181211

#lr=0.001
python kitti.py --func=train --net=hed --merge=add --check=true --augm=true --load=true --epochs=100 --lr=0.001 > ../logs/20181211/20181211_add_001.txt
mv ../weights/hed_kitti_weight_add.best.hdf5 ../weights/20181211/add_0001.hdf5

python kitti.py --func=train --net=hed --merge=avg --check=true --augm=true --load=true --epochs=100 --lr=0.001 > ../logs/20181211/20181211_avg_001.txt
mv ../weights/hed_kitti_weight_avg.best.hdf5 ../weights/20181211/avg_0001.hdf5

python kitti.py --func=train --net=hed --merge=max --check=true --augm=true --load=true --epochs=100 --lr=0.001 > ../logs/20181211/20181211_max_001.txt
mv ../weights/hed_kitti_weight_max.best.hdf5 ../weights/20181211/max_0001.hdf5

#lr=0.0001
python kitti.py --func=train --net=hed --merge=add --check=true --augm=true --load=true --epochs=100 --lr=0.0001 > ../logs/20181211/20181211_add_0001.txt
mv ../weights/hed_kitti_weight_add.best.hdf5 ../weights/20181211/add_0001.hdf5

python kitti.py --func=train --net=hed --merge=avg --check=true --augm=true --load=true --epochs=100 --lr=0.0001 > ../logs/20181211/20181211_avg_0001.txt
mv ../weights/hed_kitti_weight_avg.best.hdf5 ../weights/20181211/avg_0001.hdf5

python kitti.py --func=train --net=hed --merge=max --check=true --augm=true --load=true --epochs=100 --lr=0.0001 > ../logs/20181211/20181211_max_0001.txt
mv ../weights/hed_kitti_weight_max.best.hdf5 ../weights/20181211/max_0001.hdf5

#lr=0.00001
python kitti.py --func=train --net=hed --merge=add --check=true --augm=true --load=true --epochs=100 --lr=0.00001 > ../logs/20181211/20181211_add_00001.txt
mv ../weights/hed_kitti_weight_add.best.hdf5 ../weights/20181211/add_0001.hdf5

python kitti.py --func=train --net=hed --merge=avg --check=true --augm=true --load=true --epochs=100 --lr=0.00001 > ../logs/20181211/20181211_avg_00001.txt
mv ../weights/hed_kitti_weight_avg.best.hdf5 ../weights/20181211/avg_0001.hdf5

python kitti.py --func=train --net=hed --merge=max --check=true --augm=true --load=true --epochs=100 --lr=0.00001 > ../logs/20181211_max_00001.txt
mv ../weights/hed_kitti_weight_max.best.hdf5 ../weights/20181211/max_0001.hdf5
