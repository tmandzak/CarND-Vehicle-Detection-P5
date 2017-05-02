import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class AdvancedLaneFinding:
    def __init__(self, cal_images, cal_nx, cal_ny, test_images, minpix, usePreviousFrame, ym_per_pix, xm_per_pix):
        
        # Calibration
        self.cal_images = [mpimg.imread(img) for img in glob.glob(cal_images)] 
        self.cal_nx = cal_nx
        self.cal_ny = cal_ny
        self.findChessboardCorners()
        
        self.test_images = [mpimg.imread(img) for img in glob.glob(test_images)]
        self.img_size = (self.test_images[0].shape[1], self.test_images[0].shape[0])
        
        self.minpix = minpix
        
        # Do camera calibration given object points and image points
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_size, None, None)
        
        # Define four sided polygons for perspective transform
        self.src_poly = np.float32([[191,719],[587,454],[693,454],[1118,719]])
        self.dst_poly = np.float32([[300,719],[300,0],[1000,0],[1000,719]])
        
        self.M = cv2.getPerspectiveTransform(self.src_poly, self.dst_poly)
        self.Minv = cv2.getPerspectiveTransform(self.dst_poly, self.src_poly)
        
        self.src_poly_int = np.int32(self.src_poly).reshape((-1,1,2))
        self.dst_poly_int = np.int32(self.dst_poly).reshape((-1,1,2))
        
        self.usePreviousFrame = usePreviousFrame
        self.left_fit = []
        self.right_fit = []
        
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = ym_per_pix # meters per pixel in y dimension
        self.xm_per_pix = xm_per_pix # meters per pixel in x dimension
        
        
    # Camera calibration
    def findChessboardCorners(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.cal_ny*self.cal_nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.cal_nx,0:self.cal_ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        self.corners_images = []
        self.corners_images_failed = []

        # Step through the list and search for chessboard corners
        for img in self.cal_images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.cal_nx, self.cal_ny), None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                # Draw and display the corners
                self.corners_images.append(cv2.drawChessboardCorners(img, (self.cal_nx, self.cal_ny), corners, ret))
            else:
                self.corners_images_failed.append(img)
                
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
 
    def draw_corners_images(self, n=None):
        self._draw_images(self.corners_images, n=n)
        
    def draw_corners_images_failed(self, n=None):
        self._draw_images(self.corners_images_failed, n=n)        
        
    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    
    # Test undistortion on an image
    def draw_test_undistort(self, test_image):
        img = cv2.imread(test_image)

        dst = self.undistort(img)
        # Visualize undistortion
        self._draw_images(images=[img, dst], titles=['Original Image', 'Undistorted Image'])
        
    def _combinelists(self, l1, l2):
        res = []
        for i in range(len(l1)):
            res.append(l1[i])
            res.append(l2[i])
            
        return res    
        
    def draw_test_images_undistort(self):
        test_images = self.test_images

        dst_images = []
        for img in test_images:
            dst_images.append(self.undistort(img))
        
        self._draw_images(images=self._combinelists(test_images, dst_images), titles=['Original Image', 'Undistorted Image']*len(dst_images))
            
        return dst_images
            
    def mixed_threshold(self, img):
        # Color threshold
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  
        lower_yellow = np.array([10, 100, 100], dtype ='uint8')
        upper_yellow = np.array([30, 255, 255], dtype ='uint8')
        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  
        mask_white = cv2.inRange(img_gray, 200, 255)
        
        sbinary = cv2.bitwise_or(mask_white, mask_yellow)
        
        # Threshold x gradient
        # Sobel x
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        sxbinary = cv2.inRange(scaled_sobel, 20, 100)
        
        mixed =  cv2.bitwise_or(sbinary, sxbinary)
        
        return mixed, sbinary, sxbinary        

    def draw_test_images_color_threshold(self, test_images = []):
        if test_images == []:
            test_images = self.test_images
            
        images = []
        for img in test_images:
            images.append(self.mixed_threshold(img)[1])
                
        self._draw_images(images=self._combinelists(test_images, images), titles=['Input', 'Color thresholds']*len(images), cmap='Greys_r')
        return images
            
            
    def draw_test_images_gradient_threshold(self, test_images = []):
        if test_images == []:
            test_images = self.test_images
            
        images = []
        for img in test_images:
            images.append(self.mixed_threshold(img)[2])

        self._draw_images(images=self._combinelists(test_images, images), titles=['Input', 'Gradient thresholds']*len(images), cmap='Greys_r')
        return images
            
            
    def draw_test_images_mixed_threshold(self, test_images = []):
        if test_images == []:
            test_images = self.test_images
            
        images_color = []
        images_binary = []
        
        for img in test_images:
            mixed, sbinary, sxbinary = self.mixed_threshold(img)
            
            images_binary.append( mixed )
            images_color.append( np.dstack(( np.zeros_like(sxbinary), sxbinary, sbinary)) )
            
                
        self._draw_images(images=self._combinelists(test_images, images_color), titles=['Input', 'Thresholded Binary']*len(images_color))        
        return images_binary, images_color   
            

    def warpPerspective(self, img):
        warped = cv2.warpPerspective(img, self.M, self.img_size)
        return warped
        
    def draw_test_images_warped(self, test_images = []):
        if test_images == []:
            test_images = self.test_images
            
        src_images = []
        dst_images = []
        images_binary = []
        images_color = []
        
        for img in test_images:
            src_img = img.copy()
            if len(src_img.shape)<3:
                src_img = np.dstack((src_img, src_img, src_img))
            cv2.polylines(src_img, [self.src_poly_int], True, (255,0,0), 5)
            src_images.append(src_img)            
            
            warped = self.warpPerspective(img)
            
            dst_img = warped.copy()
            if len(dst_img.shape)<3:
                dst_img = np.dstack((dst_img, dst_img, dst_img))
            images_color.append(dst_img.copy())    
            cv2.polylines(dst_img, [self.dst_poly_int], True, (255,0,0), 5)
            dst_images.append(dst_img)
            
            if len(warped.shape)>2:
                images_binary.append(cv2.inRange(cv2.bitwise_or(warped[:,:,1], warped[:,:,2]), 1, 255))
            else:
                images_binary.append(cv2.inRange(warped, 1 , 255))

        self._draw_images(images=self._combinelists(src_images, dst_images), titles=['Input', 'Transformed']*len(src_images))     
        
        return images_binary, images_color

    def locateLaneLines(self, binary_warped):
        left_fit = self.left_fit
        right_fit = self.right_fit
        
        notFound = True
        firstRun = True
        
        while notFound & firstRun:
            if (self.usePreviousFrame == False) | (left_fit == []) | (right_fit == []):
                binary_warped = binary_warped // 255
                # Assuming you have created a warped binary image called "binary_warped"
                # Take a histogram of the bottom half of the image
                histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
                # Create an output image to draw on and  visualize the result
                out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
                # Find the peak of the left and right halves of the histogram
                # These will be the starting point for the left and right lines
                midpoint = np.int(histogram.shape[0]/2)
                leftx_base = np.argmax(histogram[:midpoint])
                rightx_base = np.argmax(histogram[midpoint:]) + midpoint

                # Choose the number of sliding windows
                nwindows = 9
                # Set height of windows
                window_height = np.int(binary_warped.shape[0]/nwindows)
                # Identify the x and y positions of all nonzero pixels in the image
                nonzero = binary_warped.nonzero()
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                # Current positions to be updated for each window
                leftx_current = leftx_base
                rightx_current = rightx_base
                # Set the width of the windows +/- margin
                margin = 100
                # Set minimum number of pixels found to recenter window
                minpix = self.minpix
                # Create empty lists to receive left and right lane pixel indices
                left_lane_inds = []
                right_lane_inds = []

                # Step through the windows one by one
                for window in range(nwindows):
                    # Identify window boundaries in x and y (and right and left)
                    win_y_low = binary_warped.shape[0] - (window+1)*window_height
                    win_y_high = binary_warped.shape[0] - window*window_height
                    win_xleft_low = leftx_current - margin
                    win_xleft_high = leftx_current + margin
                    win_xright_low = rightx_current - margin
                    win_xright_high = rightx_current + margin
                    # Draw the windows on the visualization image
                    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
                    # Identify the nonzero pixels in x and y within the window
                    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                    # Append these indices to the lists
                    left_lane_inds.append(good_left_inds)
                    right_lane_inds.append(good_right_inds)
                    # If you found > minpix pixels, recenter next window on their mean position
                    if len(good_left_inds) > minpix:
                        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                    if len(good_right_inds) > minpix:        
                        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

                # Concatenate the arrays of indices
                left_lane_inds = np.concatenate(left_lane_inds)
                right_lane_inds = np.concatenate(right_lane_inds)
                
                firstRun = False
            else:
                # Assume you now have a new warped binary image 
                # from the next frame of video (also called "binary_warped")
                # It's now much easier to find line pixels!
                nonzero = binary_warped.nonzero()
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                margin = 100
                left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
                right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  


            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds] 

            # Fit a second order polynomial to each
            try:
                left_fit = np.polyfit(lefty, leftx, 2)
                right_fit = np.polyfit(righty, rightx, 2)
                notFound = False
            except:
                left_fit, right_fit = [], []
                notFound = True
        
        self.left_fit = left_fit
        self.right_fit = right_fit
        
        if (left_fit!=[]) & (right_fit!=[]):
            #----------------------------- Offset -----------------------------------------
            x_left = left_fit[0]*719**2 + left_fit[1]*719 + left_fit[2]
            x_right = right_fit[0]*719**2 + right_fit[1]*719 + right_fit[2]
            offset = (640 - int((x_left + x_right)/2) ) * self.xm_per_pix

            #---------------------------- Curvature --------------------------------------
            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(lefty*self.ym_per_pix, leftx*self.xm_per_pix, 2)
            right_fit_cr = np.polyfit(righty*self.ym_per_pix, rightx*self.xm_per_pix, 2)
            center_fit_cr = (left_fit_cr + right_fit_cr)/2

            # Calculate the new radii of curvature
            y_eval = (binary_warped.shape[0]-1)*self.ym_per_pix

            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
            right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
            center_curverad = ((1 + (2*center_fit_cr[0]*y_eval + center_fit_cr[1])**2)**1.5) / np.absolute(2*center_fit_cr[0])
        else:
            left_curverad, right_curverad, center_curverad, offset = 0, 0, 0, 0
            left_fit, right_fit, leftx, lefty, rightx, righty = [], [], [], [], [], []
            
            
        return left_curverad, right_curverad, center_curverad, offset, left_fit, right_fit, leftx, lefty, rightx, righty


    def _draw_binary_image_lanes_located(self, binary_warped, left_fit, right_fit, leftx, lefty, rightx, righty):
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        #center_fitx = center_fit[0]*ploty**2 + center_fit[1]*ploty + center_fit[2]
        #center_fitx = (left_fitx + right_fitx)//2
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        margin = 100
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        image = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        pts = np.int32(np.round(list(zip(left_fitx, ploty))))
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img=image, pts=[pts],isClosed=False,color=(255,255,0), lineType=8, thickness = 3)

        pts = np.int32(np.round(list(zip(right_fitx, ploty))))
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img=image, pts=[pts],isClosed=False,color=(255,255,0), lineType=8, thickness = 3)  

        #pts = np.int32(np.round(list(zip(center_fitx, ploty))))
        #pts = pts.reshape((-1,1,2))
        #cv2.polylines(img=image, pts=[pts],isClosed=False,color=(255,255,0), lineType=8, thickness = 3)  
        
        return image

    def draw_binary_images_lanes_located(self, binary_images):
        images = []
        titles = []
        
        for img in binary_images:
            left_curverad, right_curverad, center_curverad, offset, left_fit, right_fit, leftx, lefty, rightx, righty = self.locateLaneLines(img)
            
            result = self._draw_binary_image_lanes_located(img, left_fit, right_fit, leftx, lefty, rightx, righty)
            
            images.append(result)
            #titles.append("Curves radiuses: "+str(int(left_curverad))+"  "+str(int(right_curverad)))
            titles.append("Lane curvature: "+str(int(center_curverad))+"  Offset: "+'{:3.2f}'.format(offset))

        self._draw_images(images=images, titles=titles)
        return images
    

    def draw_color_area_located(self, color_undist, binary_warped, left_fit, right_fit, left_curverad, right_curverad, center_curverad, offset):
        if (left_fit!=[]) & (right_fit!=[]):
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] 

            # Create an image to draw the lines on
            warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(color_warp, self.Minv, (binary_warped.shape[1], binary_warped.shape[0])) 

            # Combine the result with the original image
            image = cv2.addWeighted(color_undist, 1, newwarp, 0.3, 0)
            #text = 'L: '+str(int(left_curverad))+'m  R: '+str(int(right_curverad))+'m'
            text1 = "Lane curvature: "+str(int(center_curverad))+" m"  
            text2 = "Offset: "+'{:3.2f}'.format(abs(offset))+" m "+ ('left' if offset < 0 else 'right') + " of center"

            cv2.putText(image, text1, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            cv2.putText(image, text2, (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
        else:
            image = color_undist
        
        return image     
    
    
    def draw_test_images_area_located(self, binary_images):
        images = []
        titles = []
        
        for t_img, b_img in zip(self.test_images, binary_images):
            left_curverad, right_curverad, center_curverad, offset, left_fit, right_fit, _, _, _, _ = self.locateLaneLines(b_img)
            
            result = self.draw_color_area_located(t_img, b_img, left_fit, right_fit, left_curverad, right_curverad, center_curverad, offset)
            
            images.append(result)
            
        self._draw_images(images=images, titles=titles)

        return images    
    
    def pipeline(self, img):
        img = self.undistort(img)
        img_out = self.mixed_threshold(img)[0]
        img_out = cv2.inRange(self.warpPerspective(img_out), 1, 255)
        left_curverad, right_curverad, center_curverad, offset, left_fit, right_fit, _, _, _, _ = self.locateLaneLines(img_out)
        result = self.draw_color_area_located(img, img_out, left_fit, right_fit, left_curverad, right_curverad, center_curverad, offset)
        return result
    
    def draw_test_images_pipeline(self):
        images_processed = []
        
        for img in self.test_images:
            images_processed.append( self.pipeline(img) )
                
        self._draw_images(images=self._combinelists(self.test_images, images_processed), titles=['Input', 'Processed']*len(self.test_images))        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    