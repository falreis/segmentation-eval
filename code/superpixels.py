import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.cluster import hierarchy
from skimage.segmentation import slic, mark_boundaries, felzenszwalb
from skimage import io


def color_groundtruth(image, segments):
    #get number of segments
    n_seg = 0
    for segs in segments:
        curr_max = max(segs)
        if(curr_max > n_seg):
            n_seg = curr_max

    n_seg += 1

    #initalize variables
    colors = [[.0, .0, .0] for x in range(n_seg)]
    itens = [0 for x in range(n_seg)]
    new_image = copy.deepcopy(image)

    #indexing information
    len_seg_i = len(segments)
    len_seg_j = len(segments[0])

    for i in range(len_seg_i):
        for j in range(len_seg_j):
            index = segments[i][j]
            itens[index] += 1

    #define new values
    len_itens = len(itens)

    for i in range(len_itens):
        if(itens[i] > 0):
            colors[i] = (i*5,i*3,i*2)

    #generate new image
    for i in range(len_seg_i):
        for j in range(len_seg_j):
            index = segments[i][j]
            new_image[i][j] = colors[index]
            
    return new_image

def color_superpixel(image, segments):
    #get number of segments
    n_seg = 0
    for segs in segments:
        curr_max = max(segs)
        if(curr_max > n_seg):
            n_seg = curr_max

    n_seg += 1
    
    #initalize variables
    colors = [[.0, .0, .0] for x in range(n_seg)]
    itens = [0 for x in range(n_seg)]
    new_image = copy.deepcopy(image)

    #indexing information
    len_seg_i = len(segments)
    len_seg_j = len(segments[0])
    
    for i in range(len_seg_i):
        for j in range(len_seg_j):
            index = segments[i][j]
            colors[index] += image[i][j]
            itens[index] += 1
            
    #define new values
    len_itens = len(itens)
                    
    for i in range(len_itens):
        if(itens[i] > 0):
            colors[i] = [x / itens[i] for x in colors[i]]

    #generate new image
    for i in range(len_seg_i):
        for j in range(len_seg_j):
            index = segments[i][j]
            new_image[i][j] = colors[index]
    
    return new_image, n_seg, colors
    

def generate_ultrametric_map(blank_image, colors, segments, n_seg, step = 1, start_at = 0, stop_at = 1):
    Z = hierarchy.linkage(colors)
    
    if start_at <= 0:
        it = n_seg
    else:
        it = start_at
        
    if stop_at <= 0:
        stop_at = it - step
        
    cutz_images = []
    cutz_nsegs = []
    
    if start_at <= 0:
        cutz_images.append(mark_boundaries(blank_image, segments, color=(0, 0, 0))[:,:,0:1])
        cutz_nsegs.append(n_seg)

    for ix in range(it-step, stop_at, -step):
        cluster_size= ix #int(ix * step)
        #print(cluster_size)

        cutz = hierarchy.cut_tree(Z, n_clusters = cluster_size)
        cutz_segs = copy.deepcopy(segments)

        for i in range(len(segments)):
            for j in range(len(segments[i])):
                index = segments[i][j]
                cutz_segs[i][j] = cutz[index][0]                
                
        cutz_images.append(mark_boundaries(blank_image, cutz_segs, color=(0, 0, 0))[:,:,0:1])
        cutz_nsegs.append(cluster_size)
    
    return cutz_images, cutz_nsegs


def generate_ultrametric_image(empty_image, colors, segments, n_seg, step = 1, start_at = 0, stop_at = 1, black_color=False):
    if empty_image == None:
        empty_image = np.zeros(colors.shape,dtype=np.uint8) #create blank image to save
        
        #white image
        if black_color == False:
            empty_image.fill(255)
    
    #create hierarchy
    Z = hierarchy.linkage(colors)
    cluster_sizes = []
    
    #start value
    if start_at <= 0:
        it = n_seg
    else:
        it = start_at
    
    #stop value
    if stop_at <= 0:
        stop_at = it - step

    #first value (without ultrametric)
    if start_at <= 0:
        if black_color == True:
            color_value = 1
        else:
            color_value = 0
        
        empty_image = mark_boundaries(empty_image, segments, color=(color_value, color_value, color_value))
        cluster_sizes.append(it)
        
    #generate ultrametric values
    for ix in range(it-step, stop_at, -step):
        cluster_size= ix #int(ix * step)

        cutz = hierarchy.cut_tree(Z, n_clusters = cluster_size)
        cutz_segs = copy.deepcopy(segments)

        for i in range(len(segments)):
            for j in range(len(segments[i])):
                index = segments[i][j]
                cutz_segs[i][j] = cutz[index][0]                

        #color of the image
        if black_color == True:
            color_value = (stop_at/cluster_size)
        else:
            color_value = 1 - (stop_at/cluster_size)
                
        empty_image = mark_boundaries(empty_image, cutz_segs, color=(color_value, color_value, color_value))
        cluster_sizes.append(cluster_size)
    
    return empty_image, cluster_sizes


def process_image(image, slic_segments = 512, felz_scale = 1536, felz_min_size = 30
                  , ultrametric = True, save=False, filename = '', paths=[]
                  , ult_step = 1, ult_start_at = 0, ult_stop_at = 1):
    '''
    Process image using SLIC and Felzenszwalb algorithms
     * image: image for processing
     * slic_segments: number of slic segments
     * felz_scale: scale for felzenszwalb algorithm
     * felz_min_size: minimum size for clusters using Felzenszwalb algorithm
     * ultrametric: generate ultrametric map
     * save: save processing results
     * filename: filename with extension (only jpg)
     * paths: [0]: segmentation's path
              [1]: border's path
              [2]: ultrametric map's path
     * ult_step: ultrametric step parameter
     * ult_start_at: ultrametric start_at parameter
     * ult_stop_at: ultrametric stop_at parameter
    '''
    #process slic
    segs_slic = slic(image, n_segments = slic_segments, slic_zero = True)
    slic_image, _, _ = color_superpixel(image, segs_slic)

    #process felzenszwalb
    segs_fs = felzenszwalb(slic_image, scale = felz_scale, sigma=0.8, min_size = felz_min_size)
    fs_image, n_segs_fs, colors_fs = color_superpixel(slic_image, segs_fs)
    
    #borders
    img = np.zeros(image.shape,dtype=np.uint8) #create blank image to save
    img.fill(255)
    fs_borders = mark_boundaries(img, segs_fs, color=(0, 0, 0))
    
    #ultrametric
    if ultrametric == True:
        ultra_images, ultra_nsegs = generate_ultrametric_map(img, colors_fs, segs_fs, n_segs_fs
                                                             , ult_step, ult_start_at, ult_stop_at)
    else:
        ultra_images = None
    
    if save == True:
        if filename != '' and len(paths) == 3:
            #save segmentation image
            io.imsave((paths[0] + 'seg_' +filename), fs_image)
            
            #save borders
            io.imsave((paths[1] + 'bor_' +filename), fs_borders)

            #save ultrametric
            if ultrametric == True:
                for u_img, u_nseg in zip(ultra_images, ultra_nsegs):
                    io.imsave((paths[2] + 'ult_' + filename[:-4] + '_' + str(u_nseg) + '.jpg'), u_img)
                
    return fs_image, fs_borders, ultra_images
      