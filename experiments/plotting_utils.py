import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_grid(images, nrows, ncols):# Index for iterating over images
    pic_index = 0

    for i, image in enumerate(images[:nrows * ncols]):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off') # Don't show axes (or gridlines)

        plt.imshow(image)

    plt.show()