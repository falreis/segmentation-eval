import superpixels as sp
from skimage.segmentation import slic, mark_boundaries, felzenszwalb

def generate_boundaries(image, blank_image, method='sgb'):
    border = None
    
    if(method == 'sgb'):
        #1664;1920 | 1664;1280 | 1536;1024 | 1536;1152 | *1408;1536 | 1408;1408
        
        _, border, _ = sp.process_image(image
                                        , slic_segments = 1408
                                        , felz_scale = 1408
                                        , felz_min_size = 20
                                        , ultrametric = False
                                        , save=False)
    elif(method=='egb'):
        f_segs = felzenszwalb(image, scale=300, sigma=0.8, min_size=20)
        border = mark_boundaries(blank_image, f_segs, color=(0, 0, 0))
        
    elif(method == 'slic'):
        s_segs = slic(image, n_segments = 300, slic_zero = True) #300 ou 100
        border = mark_boundaries(blank_image, s_segs, color=(0, 0, 0))
        
    else:
        return None

    return border[:, :, 0:1]

def generate_ultrametric_image(image, blank_image, method='hsgb'):
    if(method == 'hsgb'):        
        #process slic
        segs_slic = slic(image, n_segments = 1408, slic_zero = True)
        slic_image, _, _ = sp.color_superpixel(image, segs_slic)

        #process felzenszwalb
        segs_fs = felzenszwalb(slic_image, scale = 1408, sigma=0.8, min_size = 20)
        _, n_segs_fs, colors_fs = sp.color_superpixel(slic_image, segs_fs)

        #ultrametric
        ultra_image, cluster_sizes = sp.generate_ultrametric_image(blank_image, colors_fs, segs_fs, n_segs_fs
                                            , step = 8, start_at = 48, stop_at = 1, black_color = False)
        
    elif(method=='hegb'):
        seg_felz = felzenszwalb(image, scale=300, sigma=0.8, min_size=20)
        _, n_segs, colors = sp.color_superpixel(image, seg_felz)        
        ultra_image, cluster_sizes = sp.generate_ultrametric_image(blank_image, colors, seg_felz, n_segs
                                            , step = 20, start_at = 150, stop_at = 50, black_color = False)
        
    elif(method == 'hslic'):
        seg_slic = slic(image, n_segments = 300, slic_zero = True)
        _, n_segs, colors = sp.color_superpixel(image, seg_slic)        
        ultra_image, cluster_sizes = sp.generate_ultrametric_image(blank_image, colors, seg_slic, n_segs
                                            , step = 20, start_at = 150, stop_at = 50, black_color = False)
        
    else:
        return None

    return ultra_image, cluster_sizes