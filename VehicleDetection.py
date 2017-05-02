import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VehicleDetection:
    def __init__(self, test_images, useHotMapping):
        
        self.test_images = [mpimg.imread(img) for img in glob.glob(test_images)]
        self.img_size = (self.test_images[0].shape[1], self.test_images[0].shape[0])
        
        # Define four sided polygon
        self.src_poly = np.float32([[191,719],[587,454],[693,454],[1118,719]])
        
        self.useHotMapping = useHotMapping

        
    def _draw_images(self, images, titles=[], n=None, cols=2, show_axis='on', cmap='Greys_r'):
        if len(images)>0:
            if n or n==0:
                _ = plt.imshow(images[n])
            else:    
                rows = len(images) // cols + int(bool( len(images) % cols ))

                fig, axs = plt.subplots(rows, cols, figsize=(15, rows*4))
                axs = axs.ravel()

                i = 0
                for image in images:
                    a = axs[i] #axs[i // cols, i % cols]
                    a.axis(show_axis)
                    if len(titles)==len(images):
                        a.set_title(titles[i], fontsize=20)
                    a.imshow(image, cmap)
                    i+=1
 
    def _combinelists(self, l1, l2):
        res = []
        for i in range(len(l1)):
            res.append(l1[i])
            res.append(l2[i])
            
        return res    
    
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        res = img.copy() # make a copy of the image
        for bbox in bboxes: # draw each bounding box on your image copy using cv2.rectangle()
            cv2.rectangle(res, bbox[0], bbox[1], color, thick)
        return res
    
    # Define a function to compute color histogram features  
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the RGB channels separately
        rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Generating bin centers
        bin_edges = rhist[1]
        bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return rhist, ghist, bhist, bin_centers, hist_features  
    
    def draw_color_hist(self, img):
        rh, gh, bh, bincen, feature_vec = self.color_hist(img)
    
        # Plot a figure with all three bar charts
        if rh is not None:
            fig = plt.figure(figsize=(12,3))
            plt.subplot(131)
            plt.bar(bincen, rh[0])
            plt.xlim(0, 256)
            plt.title('R Histogram')
            plt.subplot(132)
            plt.bar(bincen, gh[0])
            plt.xlim(0, 256)
            plt.title('G Histogram')
            plt.subplot(133)
            plt.bar(bincen, bh[0])
            plt.xlim(0, 256)
            plt.title('B Histogram')
            fig.tight_layout()
        else:
            print('Your function is returning None for at least one variable...')   
            
    def plot3d(self, pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
        """Plot pixels in 3D."""

        # Create figure and 3D axes
        fig = plt.figure(figsize=(8, 8))
        ax = Axes3D(fig)

        # Set axis limits
        ax.set_xlim(*axis_limits[0])
        ax.set_ylim(*axis_limits[1])
        ax.set_zlim(*axis_limits[2])

        # Set axis labels and sizes
        ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
        ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
        ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
        ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

        # Plot pixel values with colors given in colors_rgb
        ax.scatter(
            pixels[:, :, 0].ravel(),
            pixels[:, :, 1].ravel(),
            pixels[:, :, 2].ravel(),
            c=colors_rgb.reshape((-1, 3)), edgecolors='none')

        return ax  # return Axes3D object for further manipulation
        
            
    # Define a function to compute color histogram features  
    # Pass the color_space flag as 3-letter all caps string
    # like 'HSV' or 'LUV' etc.
    def bin_spatial(self, img, color_space='RGB', size=(32, 32)):
        # Convert image to new color space (if specified)
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(img)             
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(feature_image, size).ravel() 
        # Return the feature vector
        return features, feature_image        
    
    def draw_test_images_colorspaces(self, test_images=[], color_spaces = ['HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'], layer=0):
        if test_images==[]:
            test_images = self.test_images
        
        images_processed = []
                    
        for img in test_images:
            images_processed.append( img )
            for cs in color_spaces:
                res = self.bin_spatial(img, color_space=cs)[1]
                images_processed.append( res[:,:,layer] )
                #images_processed.append( (res[:,:,1]+res[:,:,2])/2 )
                
        self._draw_images(images=images_processed, titles=(['Input']+color_spaces)*len(self.test_images), cols=len(color_spaces)+1)        

    
    def pipeline(self, img):
        bboxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)), 
          ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]
        
        result = self.draw_boxes(img, bboxes)
            
        return result
    
    def draw_test_images_pipeline(self):
        images_processed = []
        
        for img in self.test_images:
            images_processed.append( self.pipeline(img) )
                
        self._draw_images(images=self._combinelists(self.test_images, images_processed), titles=['Input', 'Processed']*len(self.test_images))        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    