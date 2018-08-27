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
    @staticmethod
    def preprocess(coco, img_path, img_info, used_ids):
        """Process images into a form that can be input into Keras Models.

        Args:
            coco (COCO object): coco object of portion of the coco dataset used
            img_imnfo (dict): image to preprocess
            used_ids (array): array of ints indicating which classes of coco are being used

        Returns:
            TODO(rliaw): Should return 1 CocoImage with multiple attributes.
            bboxImgs (array): array of bounding box images for each object in the image
            object_labels (array): labels for each object in bboxImgs
            configs (array): configs for ground truth segmentation for each object
                             i.e. each config in configs is used like: show_gt_mask(*config)
            shapelyPolygons (array): Shapely Polygons of the ground truth segmentation for each object
        """
        image_url = os.path.join("../val2017/", img['file_name'])
        I = io.imread(image_url)

        # Get obj seg/bbox annotations of img
        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=used_ids, iscrowd=None)
        anns = coco.loadAnns(annIds)

        polygons, object_labels, alt_segs, colors, bboxImgs = getPolygonsLabelsMasksColorsBboxImgs(coco, I, anns)

        x = len(polygons)
        assert len(object_labels) == x, (x, len(object_labels))
        assert len(alt_segs) == x, (x, len(alt_segs))
        assert len(colors) == x, (x, len(colors))
        assert len(bboxImgs) == x, (x, len(bboxImgs))

        # In order to call show_gt_mask(*config)
        configs = []
        for i in range(len(polygons)):
            configs.append((bboxImgs[i], polygons[i], colors[i]))

        # In order to check which points are in the Shapely Polygon later
        shapelyPolygons = []
        for seg in alt_segs:
            shapelyPolygons.append(Pgon(pairwise_group(seg[0])))

        assert len(configs) == len(shapelyPolygons)

        return bboxImgs, object_labels, configs, shapelyPolygons