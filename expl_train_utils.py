import time
import numpy as np
import matplotlib.pyplot as plt

from lime import lime_image
from matplotlib.collections import PatchCollection
from shapely.geometry import mapping
from shapely.geometry import Point
from skimage.segmentation import mark_boundaries, find_boundaries


class SubPlotter:
    """Creates matplotlib images showing the image, image seg, and gt seg
    
    Note: could probably override __enter__ and __exit__ for this class
    """
        
    def __init__(self, num_rows=1, use_bandits=True):
        self.graph_count = 0
        self.use_bandits = use_bandits
        self.start = time.time()
        
    def analyzeIoU(self, model, x_train_batch, y_train_batch, shapely_polygons_batch, 
                   num_accurate=10, num_inaccurate=10, good=0.7, bad=0.3, limit=30):
        """Calculates and finds images with most accurate/inaccurate IoU
        
        Args:
            x_train_batch (np.array): array of training images
            y_train_batch (np.array): array of labels (labels are 1-hot vectors)
            shapely_polygons_batch (np.array): array of shapely polygons for each images' obj seg
            num_accurate (int): number of 'accurate' segmentations to find
            num_inaccurate (int): number of 'inaccurate' segmentations to find
            good (float): minimal IoU score to be considered an 'accurate' segmentation
            bad (float): maximal IoU score to be considered an 'inaccurate' segmentation
            limit (int): number of segmentations to consider and calcIoU for
        
        Returns:
            info (array): contains (i, expl) tuples for IoU accepted as accurate or inaccurate
                              i is the index of the image in x_train_batch
                              expl is the explanation created on the corresponding ith image
        """
        x = len(x_train_batch)
        assert len(y_train_batch) == x
        assert len(shapely_polygons_batch) == x
        i, acc, not_acc = 0, 0, 0
        info = [] # contains tuples (i, expl)
        while acc < num_accurate and not_acc < num_inaccurate or i < x: 
            explainer = lime_image.LimeImageExplainer()
            s = time.time()
            expl = explainer.explain_instance(x_train_batch[i], model.predict, top_labels=1, hide_color=0,
                                              num_samples=100, timed=True, batch_size=128,
                                              use_bandits=self.use_bandits)
            diff = time.time() - s
            s = time.time()
            score = calcIoU(shapely_polygons_batch[i], explainer.segments, explainer.features_to_use)
            print("Score: {}".format(score))
            print("Explanation time: {}".format(diff))
            print("Score time: {}".format(time.time() - s))
            if scores <= bad:
                not_acc += 1
                info.append((i, expl))
            elif scores >= good:
                acc += 1
                info.append((i, expl))
            i += 1
        return info
    
    def plot_rand_segs(self, model, used_ids, catId_to_catName, 
                       x_train_batch, y_train_batch, shapely_polygons_batch, configs_batch, num_segs=3):
        self.fig, self.axeslist = plt.subplots(ncols=3, nrows=num_segs, figsize=(8 * 3, 10 * num_segs))
        rand_selection = np.random.randint(0, high=len(x_train_batch) - 1, size=num_segs)
        
        for i in rand_selection:
            explainer = lime_image.LimeImageExplainer()
            expl = explainer.explain_instance(x_train_batch[i], model.predict, top_labels=1, hide_color=0,
                                              num_samples=100, timed=True, batch_size=128, num_features=5,
                                              use_bandits=self.use_bandits)
            # score = calcIoU(shapely_polygons_batch[i], expl)
            
            bbox_img, polygons, colors = configs_batch[i]
            
            # Display original image
            _sp = self.axeslist.ravel()[self.graph_count]
            _sp.imshow(x_train_batch[i])
            index = np.argmax(y_train_batch[i])
            catId = used_ids[index]
            catName = catId_to_catName[catId]
            _sp.set_title("True Label: {}".format(catName))
            _sp.set_axis_off()
            self.graph_count += 1
            
            # Display superpixel contrib
            temp, mask = expl.get_image_and_mask(expl.top_labels[0], positive_only=False, 
                                                 num_features=5, hide_rest=False)
            _sp = self.axeslist.ravel()[self.graph_count]
            c = mark_boundaries(temp, mask)
            outline = find_boundaries(expl.segments)
            c[outline] = (1, 1, 0)
            _sp.imshow(c / 2 + 0.5, cmap=plt.gray())
            index = np.argmax(model.predict(x_train_batch[i][np.newaxis,]))
            catId = used_ids[index]
            catName = catId_to_catName[catId]
            _sp.set_title("Predicted: {}".format(catName))
            _sp.set_axis_off()
            self.graph_count += 1
            
            # Display ground truth object segmentation
            _sp = self.axeslist.ravel()[self.graph_count]
            _sp.imshow(bbox_img)
            _sp.set_title("Image: {}".format(i))
            # _sp.set_title("Image: {}, IoU: {}".format(i, score))
            # _sp.set_axis_off()
            p = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.4)
            _sp.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=colors, linewidths=2)
            _sp.add_collection(p)
            self.graph_count += 1

    
    def plot_seg_extremes(self, info, configs_batch):
        """Plots the accurate/inaccurate predicted segmentation and ground truth segmentation
        
        Args:
            info (array): output of analyzeIoU(...) function call
            configs_batch (np.array): array of (Bounding box image, polygons, colors) tuples
        """
        self.fig, self.axeslist = plt.subplots(ncols=3, nrows=len(info), figsize=(8 * 3, 10 * len(info)))
        
        for i, expl in info:
            bbox_img, polygons, colors = configs_batch[i]
            
            # Display original image
            _sp = self.axeslist.ravel()[self.graph_count]
            _sp.imshow(bbox_img / 2 + 0.5, cmap=plt.gray())
            _sp.set_axis_off()
            self.graph_count += 1

            # Display superpixel contrib
            temp, mask = expl.get_image_and_mask(expl.top_labels[0], positive_only=False, 
                                                 num_features=5, hide_rest=False)
            _sp = self.axeslist.ravel()[self.graph_count]
            _sp.imshow(mark_boundaries(temp, mask) / 2 + 0.5, cmap=plt.gray())
            _sp.set_axis_off()
            self.graph_count += 1
            
            # Display ground truth object segmentation
            _sp = self.axeslist.ravel()[self.graph_count]
            _sp.imshow(bbox_img)
            ax = _sp.gca()
            ax.set_autoscale_on(False)
            p = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=colors, linewidths=2)
            ax.add_collection(p)
            self.graph_count += 1

    def end(self):
        """Prints out the plots and runtime"""
        plt.tight_layout()
        plt.show()
        end = time.time()
        diff = end - self.start
        print("Ran for {} seconds!! ({} minutes)".format(round(diff, 3), round(diff / 60, 3)))


def calcIoU(gt_pgon, expl):
    """Calculate IoU of gt_mask (shapely Polygon) + LIME seg
    
    Args:
        gt_pgon (shapely Polygon): mask of ground truth object segmentation
        pred_seg (np.array): segmentation algorithm output
        expl (LIME image explanation instance)
    
    Returns:
        iou (float): measure of accuracy of LIME segmentation to ground truth
    """
    pred_coords = []
    weights = expl.local_exp[expl.top_labels[0]]
    pos_segments = [x[0] for x in weights if x[1] > 0]
    for ft in pos_segments:
        shape = np.where(expl.segments == ft)
        # Not sure why transformation is required ...
        pred_coords.extend(list(zip(shape[1], shape[0] * -1 + expl.segments.shape[1])))
    
    coords_in_intersection = 0
    for coord in pred_coords:
        if gt_pgon.contains(Point(coord)):
            coords_in_intersection += 1
    print(coords_in_intersection)
    
    num_coords_in_union = len(pred_coords) + len(gt_coords) - coords_in_intersection
    
    return coords_in_intersection / num_coords_in_union 
