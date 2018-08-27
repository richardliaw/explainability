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
    def preprocess(coco, img_info, used_ids):
        pass