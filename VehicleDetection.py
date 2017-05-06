import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import hog

class VehicleDetection:
    def __init__(self, color_space, # Color space base for features
                 spatial_size, # Spatial binning dimensions
                 hist_bins, # Number of histogram bins
                 orient, # HOG orientations
                 pix_per_cell, # HOG pixels per cell
                 cell_per_block, # HOG cells per block
                 spatial_feat, # Spatial features on or off
                 hist_feat, # Histogram features on or off
                 hog_feat, # HOG features on or off
                 x_start_stop, # Min and max in x to search in slide_window()
                 y_start_stop, # Min and max in y to search in slide_window()
                 test_images, # Test images
                 train_cars, # Initialize training car images
                 train_notcars): # Initialize training non car images
        
        self.color_space = color_space 
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.x_start_stop = x_start_stop
        self.y_start_stop = y_start_stop 

        # Load test images
        self.test_images = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in glob.glob(test_images)]
        # self.img_size = (self.test_images[0].shape[1], self.test_images[0].shape[0])
        
        # Initialize training images
        self.cars = glob.glob(train_cars, recursive=True)
        self.notcars = glob.glob(train_notcars, recursive=True)
        
        
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
    
  
    # Convert image to new color space (if specified)
    # Pass the color_space flag as 3-letter all caps string
    # like 'HSV' or 'LUV' etc.    
    def convert_color(self, image, cspace='RGB'):
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'H(LS)':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:,:,1:]      
            elif cspace == 'H(L)S':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:,:,1] 
                feature_image = feature_image[:,:,None]
            elif cspace == 'HL(S)':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:,:,2]    
                feature_image = feature_image[:,:,None]
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)  
            elif cspace == '(Y)CrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)[:,:,0]
                feature_image = feature_image[:,:,None]
            elif cspace == 'Y(Cr)Cb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)[:,:,1]
                feature_image = feature_image[:,:,None]
            elif cspace == 'YCr(Cb)':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)[:,:,2]                
                feature_image = feature_image[:,:,None]
        else: feature_image = np.copy(image)

        return feature_image 
    
    def draw_test_images_colorspaces(self, test_images=[], color_spaces = ['HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'], layer=None):
        if test_images==[]:
            test_images = self.test_images
        
        images_processed = []
                    
        for img in test_images:
            images_processed.append( img )
            for cs in color_spaces:
                res = self.convert_color(img, cspace=cs)
                #res = self.bin_spatial(res)
                if layer == None:
                    images_processed.append( res )
                else:
                    images_processed.append( res[:,:,layer] )
                
        self._draw_images(images=images_processed, titles=(['Input']+color_spaces)*len(self.test_images), cols=len(color_spaces)+1)    
 
    def draw_plot3d(self, pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
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

    # Define a function to compute binned color features  
    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel() 
        # Return the feature vector
        return features    
    
    # Define a function to compute color histogram features  
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        hist_features = np.histogram(img[:,:,0], bins=nbins, range=bins_range)[0]
        for i in range(1, img.shape[2]):
            # Concatenate the histograms into a single feature vector
            hist_features = np.concatenate((hist_features, np.histogram(img[:,:,i], bins=nbins, range=bins_range)[0]))
        # Return the feature vector
        return hist_features    
    
    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, ravel=True, feature_vector = True, transform_sqrt=True):
        hog_features = []
        for channel in range(img.shape[2]):
            features = hog(img[:,:,channel], orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=transform_sqrt, block_norm='L2-Hys',
                                  visualise=False, feature_vector=feature_vector)
            hog_features.append( features )

        if ravel:
            hog_features = np.ravel(hog_features)    

        return hog_features  
    
    # Define a function to return HOG features and visualization
    def get_hog(self, img, orient, pix_per_cell, cell_per_block):
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, block_norm='L2-Hys',
                                  visualise=True, feature_vector=True)
        return features, hog_image        

    def draw_test_images_hog(self, test_images=[], cspace='H(L)S'):
        if test_images==[]:
            test_images = self.test_images
        
        images_processed = []
                    
        for img in test_images:
            img = self.convert_color(img, cspace=cspace)[:,:,0]
            images_processed.append( img )
            
            res = self.get_hog(img, self.hog_orient, self.hog_pix_per_cell, self.hog_cell_per_block)[1]
            images_processed.append( res )
                
        self._draw_images(images=images_processed, titles=['Input '+cspace, 'HOG']*len(test_images), cmap='Greys_r')    
        
    # Define a function to extract features from a single image window
    def single_img_features(self, img, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9, 
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=True, hist_feat=True, hog_feat=True):    
        #1) Define an empty list to receive features
        img_features = []
        #2) Apply color conversion if other than 'RGB'
        feature_image = self.convert_color(img, cspace=color_space)

        #3) Compute spatial features if flag is set
        if spatial_feat == True:
            spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = self.color_hist(feature_image, nbins=hist_bins)
            #6) Append features to list
            img_features.append(hist_features)
        #7) Compute HOG features if flag is set
        if hog_feat == True:
            hog_features = self.get_hog_features(feature_image, orient, pix_per_cell, cell_per_block, transform_sqrt=True)
            #8) Append features to list
            img_features.append(hog_features)

        #9) Return concatenated array of features
        return np.concatenate(img_features)
    
     # Define a function to extract features from a list of images using single_img_features
    def extract_features(self, imgs, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9, 
                            pix_per_cell=8, cell_per_block=2,
                            spatial_feat=True, hist_feat=True, hog_feat=True):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            #image = mpimg.imread(file)
            image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

            file_features = self.single_img_features(image, color_space, spatial_size,
                            hist_bins, orient, 
                            pix_per_cell, cell_per_block, 
                            spatial_feat, hist_feat, hog_feat)

            features.append(file_features)
        # Return list of feature vectors
        return features   
       
    # Define a function that takes an image,
    # start and stop positions in both x and y, 
    # window size (x and y dimensions),  
    # and overlap fraction (for both x and y)
    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None], 
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list 
    
    # Define a function to draw bounding boxes
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy    
    
    def draw_sliding_windows(self):
        image = self.test_images[0]
        
        sizes = [[64, 528], [128, 592], [192, 592], [256, 656]] 
        
        processed = []
        titles=[]
        
        for item in sizes:
            size = item[0]
            ystop = item[1]
            windows = self.slide_window(image, x_start_stop=[400, None], y_start_stop = [400, ystop], xy_window=(size, size), xy_overlap=(0.75, 0.75))
            processed.append(self.draw_boxes(image, windows))
            titles.append('Window size: '+str(size)+', Overlap: 75%')
   
        self._draw_images(images=processed, titles=titles)
    
    # Train the classifier
    def train(self):
        # Build features      
        car_features = self.extract_features(self.cars, color_space=self.color_space, 
                        spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                        orient=self.orient, pix_per_cell=self.pix_per_cell, 
                        cell_per_block=self.cell_per_block, spatial_feat=self.spatial_feat, 
                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)
              
        notcar_features = self.extract_features(self.notcars, color_space=self.color_space, 
                        spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                        orient=self.orient, pix_per_cell=self.pix_per_cell, 
                        cell_per_block=self.cell_per_block, spatial_feat=self.spatial_feat, 
                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)      

    def pipeline(self, img):
       

            
        return 0
    
    def draw_test_images_pipeline(self):
        images_processed = []
        
        for img in self.test_images:
            images_processed.append( self.pipeline(img) )
                
        self._draw_images(images=self._combinelists(self.test_images, images_processed), titles=['Input', 'Processed']*len(self.test_images))        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    