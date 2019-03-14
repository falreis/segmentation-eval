# Combining convolutional side-outputs for image segmentation and border detection

## Download Datasets

Download the databases by following the tutorials for each one:
* [BSDS500](data/BSDS)
* [KITTI](data/Kitti/) 

You can only download the dataset you will work with. Follow the correct instructions and keep folder structure to avoid change the code in the first tests.

## Folder description:

The folders and files are described below:

- [bsds](bsds/) contains the source code for BSDS500 dataset *(under development)*;
- [data](data/) folder with original datasets;
- [datasets](datasets/) preprocessed datasets (used for the algorithms);
- [export](export/) export files to evaluate performance of the algorithm;
- [kitti](kitti/) contains the source code for Kitti dataset;
- [results](results/)
- [weights](weights) contains the best weights achieved;
- [helper.py](helper.py) code to performance one-hot-encoded, used by the algorithm;
- [i2dl.yml](i2dl.yml) contais Anaconda environment file. Check Anaconda's [help](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html);
- [load_weights.py](load_weights.py) ;
- [vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5](vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5) ;
- [visualize.py](visualize.py) ;
