from mrcnn.config import Config
from mrcnn import utils
from tqdm import tqdm
import numpy as np
import cv2
import mrcnn.model as modellib



class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    #moze to zwiekszyc

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
    #STEPS_PER_EPOCH = 5

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    #VALIDATION_STEPS = 1


class BrainDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, nums, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "roi")
        self.add_class("shapes", 2, "other_hemisphere")
        self.add_class("shapes", 3, "na")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in tqdm(nums):
            bg_color, shapes = self.random_image(i)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        #image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        #image = image * bg_color.astype(np.uint8)
        if getattr(self, 'root_path') is None:
            self.root_path = '/mnt/sdb1/neuroinf/mask_rcnn_fork/Mask_RCNN/datasets/nucleus/brain/'
        #for shape, color, dims in info['shapes']:
            #image = self.draw_shape(image, shape, dims, color)
        image = cv2.imread('%s/%i/images/%i.png' %(self.root_path, image_id, image_id))
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        if getattr(self, 'root_path') is None:
            self.root_path = '/mnt/sdb1/neuroinf/mask_rcnn_fork/Mask_RCNN/datasets/nucleus/brain/'
        #root_path = '/mnt/sdb1/neuroinf/mask_rcnn_fork/Mask_RCNN/datasets/nucleus/brain/'
        img = cv2.imread('%s/%i/masks/%i.png' %(self.root_path, image_id, image_id))[:,:,0]
        msk = img == 255
        #print(img.max())
        img[msk] = 2
        #print (img)
        msk =  img >= 10
        img[msk] = 1
        #print(np.nonzero(img))
        mask = np.zeros([img.shape[0], img.shape[1], count], dtype=np.uint8)
        #print(mask.shape)
        for i in range(len(info['shapes'])):
            #print(i)
            msk = img == i+1
            img_copy = img.copy()
            img_copy = np.zeros([img.shape[0], img.shape[1],1],dtype=np.uint8)
            img_copy[msk] = 1
            #print(np.nonzero(img_copy)[0].shape)
            mask[:, :, i:i+1] = img_copy

        '''
        msk = mask == 177 * 2
        mask[msk] = 2
        print (mask)
        msk = 176 >= mask
        mask[msk] = 1
        print(np.nonzero(mask))
        '''
        
        '''
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        '''
        #print(shapes)
        #print('mask:')
        #print(mask.shape)
        class_ids = np.array([self.class_names.index(s) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'roi':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "other_hemisphere":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "na":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["roi", "other_hemisphere", "na"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, i):#height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        bg_color = np.array([0 for _ in range(3)])
        shapes = ['roi', 'other_hemisphere', 'na']
        return bg_color, shapes


# Training dataset
def read_datasets():
    dataset_train = BrainDataset()
    dataset_train.root_path = '/mnt/sdb1/neuroinf/mask_rcnn_fork/Mask_RCNN/datasets/nucleus/brain/'
    #dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    height, width = cv2.imread('/mnt/sdb1/neuroinf/mask_rcnn_fork/Mask_RCNN/datasets/nucleus/brain/0/images/0.png').shape[:2]
    dataset_train.load_shapes(range(0,404,2), height, width)#config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()
    # Validation dataset
    dataset_val = BrainDataset() #bedzie trzeba to podzielic jakos
    dataset_val.root_path = '/mnt/sdb1/neuroinf/mask_rcnn_fork/Mask_RCNN/datasets/nucleus/brain/'
    #dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.load_shapes(range(1, 404, 2), height, width)
    dataset_val.prepare()
    # Validation dataset
    dataset_val_14 = BrainDataset() #bedzie trzeba to podzielic jakos
    #dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    height, width = cv2.imread('/mnt/sdb1/neuroinf/mask_rcnn_fork/Mask_RCNN/datasets/nucleus/brain/0/images/0.png').shape[:2]
    dataset_val_14.root_path = '/mnt/sdb1/neuroinf/mask_rcnn_fork/Mask_RCNN/datasets/nucleus/brain_query/'
    dataset_val_14.load_shapes(range(0, 404), height, width)
    dataset_val_14.prepare()
    return dataset_train, dataset_val, dataset_val_14


def get_model(init_with = "last", config="", MODEL_DIR = "", COCO_MODEL_PATH=""):
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
    # Which weights to start with?
    init_with = "last"  # imagenet, coco, or last
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
    return model


def train(model=None, no_epochs=[], dataset_train=None, dataset_val=None,
          MODEL_DIR=None,config =None):
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    #no_epochs = [4,8]
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=no_epochs[0], 
            layers='heads')
    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also 
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=no_epochs[1], 
            layers="all")
    # Save weights
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
    model.keras_model.save_weights(model_path)
    # /mnt/sdb1/neuroinf/mask_rcnn_fork/Mask_RCNN/logs/shapes20190325T1726/mask_rcnn_shapes_0008.h5
    return model


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def load_inference_model(inference_config=None, MODEL_DIR=None): 
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)
    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()
    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model
