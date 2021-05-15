# Basic packages 


import numpy as np 
import scipy as scipy 
import os
import sys
import json
import datetime
import skimage.draw
import cv2
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa # For image augmentation


#%cd drive/MyDrive/
# To find the path for Mask_RCNN
sys.path.insert(1, 'drive/MyDrive/Mask_RCNN')
from mrcnn import utils


# ========================================================================
# ========================================================================

class GhostsDataset(utils.Dataset):
 
    def load_ghosts(self, dataset_dir, subset):
        """Load a subset of the ghosts dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have three classes to add.
        self.add_class("Type", 1, "Bright")
        self.add_class("Type", 2, "Faint")
        self.add_class("Type", 3, "Rays")
 
        # Train or validation dataset?
        assert subset in ["Training_set", "Validation_set", "Test_set"]
        dataset_dir = os.path.join(dataset_dir, subset)
 
        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations1 = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))

        
        # print(annotations1)
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
    
        annotations = [a for a in annotations if a['regions']]



 
        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['Type'] for s in a['regions']]
            #print("objects:",objects)
            name_dict = {"Bright": 1,"Faint": 2,"Rays":3}
            # key = tuple(name_dict)
            num_ids = [name_dict[a] for a in objects]
 
            # num_ids = [int(n['Event']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. 
            #print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)#/255.0
            height, width = image.shape[:2]
 
            self.add_image(
                "Type",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)
 
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a bottle dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "Type":
            return super(self.__class__, self).load_mask(image_id)
 
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        #print(info["source"])
        if info["source"] != "Type":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
      
        for i, p in enumerate(info["polygons"]):
            name = p['name']
            if (name=='polyline'):
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                a = (rr<400)&(cc<400)
                mask[rr[a], cc[a], i] = 1
            elif (name=='polygon'):
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                a = (rr<400)&(cc<400)
                mask[rr[a], cc[a], i] = 1
            elif (name=='rect'):
                rr, cc = skimage.draw.rectangle(start=(p['y'], p['x']),extent=(p['height'],p['width']))
                a = (rr<400)&(cc<400)
                mask[rr[a], cc[a], i] = 1
            elif (name=='circle'):
                rr, cc = skimage.draw.circle(p['cy'],p['cx'],p['r'])
                a = (rr<400)&(cc<400)
                mask[rr[a], cc[a], i] = 1
            else: #This is the case when you have an ellipse 
                rr, cc = skimage.draw.ellipse(p['cy'],p['cx'],p['ry'],p['rx'],shape=None,rotation=-p['theta'])
                a = (rr<400)&(cc<400)
                mask[rr[a], cc[a], i] = 1
            
 
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids
 
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Type":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
