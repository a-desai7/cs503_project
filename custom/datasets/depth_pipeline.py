import mmcv
import numpy as np
import os

from mmseg.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadDepthFromFile(object):
    def __init__(self, depth_root=None, depth_suffix='.png'):
        self.depth_root = depth_root
        self.depth_suffix = depth_suffix

    def __call__(self, results):
        # Get image filename (absolute or relative)
        img_filename = results['img_info']['filename'] if 'img_info' in results else results['filename']
        # Get basename (e.g. 0001.png)
        basename = os.path.basename(img_filename)
        # Compose depth path
        if self.depth_root is not None:
            depth_filename = os.path.join(self.depth_root, basename)
        else:
            # fallback: replace 'img' with 'depth' in path
            depth_filename = img_filename.replace('img', 'depth')
        if not depth_filename.endswith(self.depth_suffix):
            depth_filename = depth_filename.rsplit('.', 1)[0] + self.depth_suffix
        if not os.path.exists(depth_filename):
            raise FileNotFoundError(f"Depth file not found: {depth_filename} for image {img_filename}")
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
