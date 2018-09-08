import numpy as np
import os
import skimage.io as io
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon as Pgon


def show_gt_mask(bbox_img, polygons, colors):
    """Show ground truth object segmentation mask
    
    Args:
        bbox_img (array): cropped version of original image to bounding box of object segmentation mask
        polygons (array): array of matplotlib Polygon objects
        colors (array): array of floats representing colors
    
    Returns:
        None
        
    Output:
        Shaded and outlined object segmentation mask in the bounding box image provided
    """
    assert len(polygons) == len(colors)
    plt.imshow(bbox_img)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    p = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=colors, linewidths=2)
    ax.add_collection(p)
    
def pairwise_group(alt_seg):
    """Converts a flattened array to an array of tuples
    
    Args:
        alt_seg (array-like): array of sequential object segmentation points
    
    Returns:
        pgon_pts (array of tuples): array of sequential points of a polygon
    """
    pgon_pts = []
    assert len(alt_seg) % 2 == 0
    for i in range(len(alt_seg) // 2):
        pgon_pts.append((alt_seg[2 * i], alt_seg[2 * i + 1]))
    return pgon_pts

def getBboxImgXY(img, ann, margin=0.2):
    """Calculates the bounding box subimage of the object in the annotation
    
    Args:
        img (np.array): original image
        ann (dict): an annotation provided by the Coco dataset
        margin (float)
    Returns:
        bboxImg (np.array): bounding box subimage on object
        x0 (int): minimal x-coordinate of the bounding box
        y0 (int): minimal y-coordinate of the bounding box
    """
    assert 0 <= margin < 1.0, "Margin is out of band!"
    def validate(x0, x1, y0, y1):
        y0 = int(max(y0, 0))
        y1 = int(min(y1, img.shape[0]))
        
        x0 = int(max(x0, 0))
        x1 = int(min(x1, img.shape[1]))
        return x0, x1, y0, y1
    
    x0, y0, width, height = ann['bbox']
    maxmargin = max(width, height) * (1 + margin)
    x_mid = (x0 + width/2)
    y_mid = (y0 + height/2)
    x1, y1 = x_mid + maxmargin/2, y_mid + maxmargin/2
    x0, y0 = x_mid - maxmargin/2, y_mid - maxmargin/2
    
    x0, x1, y0, y1 = validate(x0, x1, y0, y1)
    bboxImg = np.copy(img[y0:y1, x0:x1])
    return bboxImg, x0, y0

def getPgonsBboxImgsColorsAltSegs(img, anns, polygons):
    """Retrieve polygons, bounding box images, colors, and segmentation coordinates of objects
    
    Args:
        img (np.array): image to process
        anns (array): array of annotations for objects
        polygons (array): array of matplotlib Polygons (for original image)
    
    Returns:
        pgons (array): array of matplotlib Polygon objects (for bbox image)
        bboxImgs (array): array of bounding box subimages of each object in the image    
        colors (array): colors to use to display visuals
        alt_segs (array): array containing new object seg coordinates relative to bboxImg
    """
    assert len(anns) == len(polygons)
    pgons, bboxImgs, colors, alt_segs = [], [], [], []
    for i in range(len(anns)):
        ann = anns[i]
        
        bboxImg, x0, y0 = getBboxImgXY(img, ann)
        bboxImgs.append(bboxImg)

        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        if 'segmentation' in ann:
            if type(ann['segmentation']) == list:
                
                alt_seg = np.copy(ann['segmentation'])
                for seg in alt_seg:
                    for i in range(len(seg)):
                        if i % 2 == 0:
                            seg[i] = int(seg[i] - x0)
                        else:
                            seg[i] = int(seg[i] - y0)
                alt_segs.append(alt_seg)

                _pgons = []
                _colors = []
                for seg in alt_seg:
                    poly = np.array(seg).reshape((int(len(seg)/2), 2))
                    _pgons.append(Polygon(poly))
                    _colors.append(c)
                pgons.append(_pgons)
                colors.append(_colors)
    return pgons, bboxImgs, colors, alt_segs

def getPolygonsLabelsMasksColorsBboxImgs(coco, img, anns):
    """Retrieve coordinates, labels, gt mask, and bounding box of objects in an image
    
    Args:
        coco (COCO object): coco object of portion of the dataset used
        img (np.array): image to process
        anns (array): array of annotation provided for the image by the dataset
        
    Returns:
        pgons (array): array of matplotlib Polygon objects
        object_labels (array): labels (int) for each object
        alt_segs (array): array containing new object seg coordinates relative to bboxImg
        colors (array): colors to use to display visuals
        bboxImgs (array): array of bounding box subimages of each object in the image    
    """
    polygons, object_labels = [], []
    _to_remove = [] # Contains annotations without object segmentations
    
    for ann in anns:
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        if 'segmentation' in ann:
            if type(ann['segmentation']) == list:
                # Get label info
                cat_info = coco.loadCats(ann['category_id'])[0]
                object_labels.append((cat_info['id'], cat_info['name']))
                
                # Create matplotlib Polygon
                polys = []
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg)/2), 2))
                    polys.append(Polygon(poly))
                polygons.append(polys)
            else:
                _to_remove.append(ann)
        else:
            _to_remove.append(ann)
    
    for a in _to_remove:
        anns.remove(a)
    
    pgons, bboxImgs, colors, alt_segs = getPgonsBboxImgsColorsAltSegs(img, anns, polygons)
    return pgons, object_labels, alt_segs, colors, bboxImgs

# Get categories with largest number of images to use
def get_largest_categories(coco, num_categories):
    """
    Returns:
        list of {'supercategory': 'person', 'id': 1, 'name': 'person'}"""
    img_per_category = [(i, len(coco.getImgIds(catIds=i))) for i in range(90)]
    img_per_category.sort(key=lambda x: x[1])
    usedCatImgs = img_per_category[-num_categories:]  # list of (catId, numImgs) tuples

    # number of images available for smallest used cat(egory)
    minNumImgs = usedCatImgs[0][1]
    used_ids = [tup[0] for tup in usedCatImgs]  # list of catIds used

    used_categories = coco.loadCats(coco.getCatIds(catIds=used_ids))
    cat_names = [cat['name'] for cat in used_categories]
    print('{} COCO categories used: \n{}\n'.format(
        len(used_categories), ' '.join(cat_names)))
    return used_categories



class CocoImage():
    def __init__(self, image, label, config, polygon):
        self.data = image
        self.label = label
        self.config = config
        self.polygon = polygon
        
    @staticmethod
    def preprocess(coco, directory, all_img_info, used_ids, filter_fn=lambda x: True):
        """
        
        Returns:
            list of CocoImage objects for each object segment"""
        all_coco_imgs = []
        for img_info in all_img_info:
            I = io.imread(os.path.join(directory, img_info["file_name"]))

            # Get obj seg/bbox annotations of img
            annIds = coco.getAnnIds(
                imgIds=img_info['id'], catIds=used_ids, iscrowd=None)
            anns = coco.loadAnns(annIds)

            all_subimage_data = getPolygonsLabelsMasksColorsBboxImgs(coco, I, anns)
            assert all(len(ret) == len(all_subimage_data[0]) for ret in all_subimage_data)

            polygons, object_labels, alt_segs, colors, bboxImgs = all_subimage_data

            # In order to call show_gt_mask(*config)
            configs = list(zip(bboxImgs, polygons, colors))

            # In order to check which points are in the Shapely Polygon later
            shapelyPolygons = [Pgon(pairwise_group(seg[0])) for seg in alt_segs]
            all_coco_imgs += [CocoImage(*args) for args in 
                              zip(bboxImgs, object_labels, configs, shapelyPolygons)
                              if filter_fn(args[0])]
        return all_coco_imgs
        
        
        
#     @staticmethod
#     def _preprocess(coco, img_path, img_info, used_ids):
#         """Process images into a form that can be input into Keras Models.

#         Args:
#             coco (COCO object): coco object of portion of the coco dataset used
#             img_imnfo (dict): image to preprocess
#             used_ids (array): array of ints indicating which classes of coco are being used

#         Returns:
#             TODO(rliaw): Should return 1 CocoImage with multiple attributes.
#             bboxImgs (array): array of bounding box images for each object in the image
#             object_labels (array): labels for each object in bboxImgs
#             configs (array): configs for ground truth segmentation for each object
#                              i.e. each config in configs is used like: show_gt_mask(*config)
#             shapelyPolygons (array): Shapely Polygons of the ground truth segmentation for each object
#         """
#         I = io.imread(img_path)

#         # Get obj seg/bbox annotations of img
#         annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=used_ids, iscrowd=None)
#         anns = coco.loadAnns(annIds)

#         polygons, object_labels, alt_segs, colors, bboxImgs = getPolygonsLabelsMasksColorsBboxImgs(coco, I, anns)

#         x = len(polygons)
#         assert len(object_labels) == x, (x, len(object_labels))
#         assert len(alt_segs) == x, (x, len(alt_segs))
#         assert len(colors) == x, (x, len(colors))
#         assert len(bboxImgs) == x, (x, len(bboxImgs))

#         # In order to call show_gt_mask(*config)
#         configs = []
#         for i in range(len(polygons)):
#             configs.append((bboxImgs[i], polygons[i], colors[i]))

#         # In order to check which points are in the Shapely Polygon later
#         shapelyPolygons = []
#         for seg in alt_segs:
#             shapelyPolygons.append(Pgon(pairwise_group(seg[0])))

#         assert len(configs) == len(shapelyPolygons)

#         return bboxImgs, object_labels, configs, shapelyPolygons