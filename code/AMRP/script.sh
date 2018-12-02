python gen_model-kitti.py --merge=maj --vote=2
python run-kitti.py --net=hed --merge=maj --vote=2 --check=true > ../logs/20181130_maj2.txt
mv ../weights/hed_kitti_weight_maj_2.best.hdf5 ../weights/maj2.hdf5

python gen_model-kitti.py --merge=maj --vote=3
python run-kitti.py --net=hed --merge=maj --vote=3 --check=true > ../logs/20181130_maj3.txt
mv ../weights/hed_kitti_weight_maj_3.best.hdf5 ../weights/maj3.hdf5

python gen_model-kitti.py --merge=max
python run-kitti.py --net=hed --merge=max --check=true > ../logs/20181130_max.txt
mv ../weights/hed_kitti_weight_max.best.hdf5 ../weights/max.hdf5

python gen_model-kitti.py --merge=add
python run-kitti.py --net=hed --merge=add --check=true > ../logs/20181130_add.txt
mv ../weights/hed_kitti_weight_add.best.hdf5 ../weights/add.hdf5

python gen_model-kitti.py --merge=avg
python run-kitti.py --net=hed --merge=avg --check=true > ../logs/20181130_avg.txt
mv ../weights/hed_kitti_weight_avg.best.hdf5 ../weights/avg.hdf5
