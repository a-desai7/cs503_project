import mmcv
import numpy as np
import os

from mmseg.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadDepthFromFile(object):
    def __init__(self, depth_suffix='.png', depth_folder=None):
        self.depth_suffix = depth_suffix
        self.depth_root = depth_folder if depth_folder is not None else 'depth'

    def __call__(self, results):
        img_path = results['img_full_path']
        image_name = os.path.basename(img_path)
        image_dir = os.path.dirname(img_path)
        image_dir_name = os.path.basename(image_dir)
        data_dir = os.path.dirname(image_dir)
        
        # The depth directory is either in the directory name replaced with 'depth' or appended by "_depth"
        # Check first to see if a directory exists thats just "depth" in the name
        if os.path.exists(os.path.join(data_dir, self.depth_root)):
            depth_directory = os.path.join(data_dir, self.depth_root)
        elif os.path.exists(os.path.join(data_dir, image_dir_name + '_depth')):
            depth_directory = os.path.join(data_dir, image_dir_name + '_depth')
        else:
            raise FileNotFoundError(f"Depth directory not found for image {img_path}")
        
        # Construct the depth filename
        depth_filename = os.path.join(depth_directory, image_name)        

        if not depth_filename.endswith(self.depth_suffix):
            depth_filename = depth_filename.rsplit('.', 1)[0] + self.depth_suffix
        if not os.path.exists(depth_filename):
            raise FileNotFoundError(f"Depth file not found: {depth_filename} for image {img_path}")
        depth = mmcv.imread(depth_filename, flag='grayscale')
        depth = depth.astype(np.float32) / 255.0  # normalize to [0, 1]
        results['depth'] = depth
        results['depth_filename'] = depth_filename
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(depth_root={self.depth_root})"

@PIPELINES.register_module()
class ConcatDepth(object):
    def __call__(self, results):
        img = results['img']
        depth = results['depth']
        if len(depth.shape) == 2:
            depth = depth[..., None]
        img_depth = np.concatenate([img, depth], axis=2)
        results['img'] = img_depth
        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'
