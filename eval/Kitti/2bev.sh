#####################################################
### TEST NO MORPHOLOGY
#####################################################

cd ../../code/export/Kitti/20190102/alo/avg/test
rename 's/umm_road_/umm_/' * 
rename 's/um_road_/um_/' * 
rename 's/uu_road_/uu_/' *

rename 's/umm_/umm_road_/' * 
rename 's/um_/um_road_/' * 
rename 's/uu_/uu_road_/' *

cd ../../../../../../../eval/Kitti

python transform2BEV.py '../../code/export/Kitti/20190102/alo/avg/test/*.png' '../../code/datasets/Kitti/data_road/testing/calib' '../../code/export/Kitti/20190102/alo/avg/bev_test'

#####################################################
### TEST MORPHOLOGY
#####################################################

cd ../../code/export/Kitti/20190102/alo/avg/test_morf
rename 's/umm_road_/umm_/' * 
rename 's/um_road_/um_/' * 
rename 's/uu_road_/uu_/' *

rename 's/umm_/umm_road_/' * 
rename 's/um_/um_road_/' * 
rename 's/uu_/uu_road_/' *

cd ../../../../../../../eval/Kitti

python transform2BEV.py '../../code/export/Kitti/20190102/alo/avg/test_morf/*.png' '../../code/datasets/Kitti/data_road/testing/calib' '../../code/export/Kitti/20190102/alo/avg/bev_test_morf'

#####################################################
### OTHER
#####################################################

#python simpleExample_transformTestResults2BEV.py '../../code/export/Kitti/20190102/alo/avg/bev_test_morf'
