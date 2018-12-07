cd ../../code/export/Kitti/hed/train
rename 's/umm_/umm_road_/' * 
rename 's/um_/um_road_/' * 
rename 's/uu_/uu_road_/' *

cd ../../../../../eval/Kitti/
python evaluateRoad.py ../../code/export/Kitti/hed/train ../../code/export
