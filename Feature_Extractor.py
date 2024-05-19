import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import cv2
import SimpleITK as sitk
from radiomics import featureextractor
import tables as tb

TOTAL_SLICES = 155
TOTAL_VOLUME = 370

RADIOMIC_FEATURES = [ # Shape Features
                  'original_shape_MajorAxisLength', 'original_shape_MinorAxisLength', 'original_shape_Elongation',
                  'original_shape_LeastAxisLength', 'original_shape_Flatness', 'original_shape_Maximum2DDiameterRow',
                  'original_shape_MeshVolume', 'original_shape_VoxelVolume', 'original_shape_Maximum2DDiameterSlice',
                  'original_shape_Maximum3DDiameter', 
                  # intensity Features
                  'original_firstorder_Uniformity', 'original_firstorder_Range', 
                  'original_firstorder_Maximum', 'original_firstorder_InterquartileRange', 'original_firstorder_90Percentile',
                  'original_firstorder_Kurtosis', 'original_firstorder_RootMeanSquared', 'original_firstorder_MeanAbsoluteDeviation',
                  'original_firstorder_Mean', 'original_firstorder_Median',
                  # texture Features
                  'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Id', 'original_glcm_Idm', 
                  'original_gldm_GrayLevelNonUniformity', 'original_gldm_LargeDependenceEmphasis', 'original_glrlm_RunEntropy',
                  'original_glcm_MaximumProbability', 'original_glcm_JointEnergy', 'original_glrlm_RunLengthNonUniformityNormalized'
                  ]


class Extractor():
    def __init__(self, path, volume_index) -> None:
        mask_list = []
        image_list = []
        for i in range(0, TOTAL_SLICES):
            file_name = 'volume_{}_slice_{}.h5'.format(str(volume_index), str(i))        
            full_path = path + '/' + file_name 
            #print(full_path)
            with tb.open_file(full_path, mode='r') as h5:
                mask_list.append(h5.root.mask.read())
                image_list.append(h5.root.image.read())
        
        self.image = np.array(image_list)
        self.mask = np.array(mask_list)
        self.volume_index = volume_index
        print(self.image.shape)

    # Function to create a merged mask from the list of masks uploaded from the dataset
    # returns the merged mask
    def merge_masks2(self, mask_array):
        # merge non-overlapping masks by addition
        merge_one = mask_array[:,:,0] + mask_array[:,:,1] + mask_array[:,:,2]
        return merge_one
    
    # Function to determine the maximum tumour area present in a slice of an idividual MRI Volume 
    def Maximum_tumout_area(self):
        # set the number  of slices in a volume, set an empty list to accumulate all the glioma volume in voxels
        no_slices=self.mask.shape[0]
        slices_count= []

        # Loop over all slices calulating the glioma volume in each slice
        for slice in range(no_slices):
            #set the slice count to zero
            slice_count = 0

            # Obtain the image of the three masks collectively containing the whole tumour in one image 
            merged_mask = self.merge_masks2(self.mask[slice,:,:,:])
            voxel_count = np.count_nonzero(merged_mask > 0) #calculate the number of  non zero voxels in the image of the glioma

            #Append the voxel count to a list
            slices_count.append(voxel_count)
        return slices_count



    # Determine the maximum diameter of the tumour in a volume using Principle Component Analysis (PCA)
    # take as a parameter all 3 masks dowloaded from Kaggle and returns a list of the diqmeters of the tumor for each slice.
    def tumour_diameter(self):
        # create a PCA object (no of components is one which returns the variance for only one eigenvector), 
        # and list to store the tumor diameter for a slice
        glioma_pca = PCA(n_components=1)
        slices_diameter = []
        
        # loop to determine the standard deviation, from PCA, as an approximation to the radius for eaxch slice of a volume
        for slice in range(TOTAL_SLICES):
        
            # create list to transform x and y co-ordinates in the mask image (mask image is binarized) 
            # to a list of x and y co-ordinates of pixels in the image having a value of 1.
            mask_newaxis = []
        
            # call merge_mask2 to return an image of the overlayed masks
            merged_mask = self.merge_masks2(self.mask[slice,:,:,:])
        
            # Transform the x and y coordinates in the image into a list of lists where the nested list contains the x any y 
            # co-ordinates in the image of all pixels whose value is 1.
            for x in range(merged_mask.shape[0]):
                for y in range(merged_mask.shape[1]):
                    if merged_mask[x,y] == 1:
                        mask_newaxis.append([x,y])
        
            # if no tumour is detected in the mask (ie no value of 1) then append 0 to the list of radius length for that slice
            if len(mask_newaxis) == 0:
                slices_diameter.append(0)
                continue
        
            # fit the tranformed x and y co-ordinates to the PCA object and get the variance
            glioma_pca.fit(mask_newaxis)
            variance = glioma_pca.explained_variance_
        
            # Append the diameter to the diameters list as twice the radius
            slices_diameter.append((variance[0]**0.5)*2)

        return slices_diameter



    # Function to transform the list of x and y co-ordinates returned by the openCV findContours method
    # into their corresponding pixels on an image, ie create a new image wioth the contour displayed in 2D
    def transform_contour(self, contours):
        # Set the width and height of the image to be created, and create a blank image of zeroes
        image_w=240
        image_h=240
        contour_image = np.zeros((image_w,image_h))

        # Loop over all the x and y co-ordinates in the co-ordinate list, Set the corresponding x and y
        # co-ordinates to 255, ie produce a white contour on a black background (binary image)
        for contour in range(len(contours[0])):
            contour_image[contours[0][contour][0][1],contours[0][contour][0][0]]=255
        
        # Return the image of the contour
        return contour_image

    # Function to produce a contour 5 pixels in width, of the brain (cerebral cortex), by producing a contour of the brain
    # within a slice; storing the image of the contour; then subtracting this contour from the original image and producing
    # another contour from the reduced image; then adding this image to the original contour. This process is repeated unitl 
    # the image of the stored contours consists of 5 contours, hence being 5 pixels wide,
    def merge_contours(self, image):
        #set variables for the width and height of the image, and the number or layers required for the combined contour
        image_w=240
        image_h=240
        outer_layers = 5

        # Convert the greyscale image of the brain into a binarized format, with all values above zero
        # set to 255 ie the brain matter will turn completeley white creating a sharp contrast at the 
        # border.
        ret, image_th = cv2.threshold(image,0,255, cv2.THRESH_BINARY)

        # convert the image to the correct datatype and create a blank image to store the created contours
        image_uint8 = image_th.astype(np.uint8)
        collected_contours = np.zeros((image_w,image_h))

        
        # Loop to calculate the contours of the brain image (outer contour); accumlate the contours in the
        # one image; subtract the produced contour form the image and repeat the procdedure using the reduced image
        # until 5 layers of contours have been added.
        for cont_num in range(outer_layers):
            # method to return the contours from the openCV findContours method. The parameter RETR_EXTERNAL ensure
            # only co-ordinates of the pixels in the outer contour of the brain image are returned. If no contour is found
            # there is no tumor in the slice the the function continues to process the next slice.
            contours, _ = cv2.findContours(image_uint8,cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0: continue

            # Call the transform_contours function to return an image of the contour.
            contour_image = self.transform_contour(contours)

            # Add the image of the contour to the blank image and accumulate the remaining 4 cnotours on subsequent loop iterations
            collected_contours += contour_image

            # subtract the contour from the image of the brain to produce a new image reduced in size by one pixel around the edge/perimeter
            # and change the datatype to uint8 to promote continued processing of the image
            reduced_image = image_uint8 - contour_image
            image_uint8 = reduced_image.astype(np.uint8)

        # return the combined image of the contours
        return collected_contours



    # function to determine the maximum volume of the Glioma within a MRI volume, by using a combined mask image of the Glioma
    # as a blocking/filtering mask on an image of the cerrebral cortex. 
    def glioma_cortex_invasion(self):
        # Set the variable to accumulate the numebr of voxels invaded by the glioma in the cerebral cortex across the MRI volume 
        total_voxels = 0

        # loop to process each slice in an MRI volume calculating the glioma occupied part ot the cerebral cortex for each slice
        for slice in range(self.image.shape[0]):
            T1_native = self.merge_contours(self.image[slice,:,:,1]) # get the 5 pixel thick contour of the brain from a slice
            glioma_merged_mask = self.merge_masks2(self.mask[slice,:,:,:]) # Get the image of the merged masks
            glioma_overlap = cv2.bitwise_and(T1_native.astype(np.uint8), glioma_merged_mask) # Return the overlap of the glioma and the cerebral cortex
            voxel_count = np.count_nonzero(glioma_overlap == 1) # calculate the nuber of voxels labelled 1 (the overlap)
            total_voxels += voxel_count
        
        return total_voxels
        


    def radiomic_features(self):

        features_sets = []
        for slice in range(4):
            # Convert the NumPy array to a SimpleITK image
            image_array = self.image[:,:,:,slice]
            image = sitk.GetImageFromArray(image_array)

            # Load the segmentation mask using SimpleITK
            
            merged_mask = np.sum(self.mask, axis=-1)
            mask = sitk.GetImageFromArray(merged_mask)

            # Configure the PyRadiomics feature extractor using default parameters
            extractor = featureextractor.RadiomicsFeatureExtractor()
            # Extract radiomic features
            features = extractor.execute(image, mask)
            features_sets.append(features)
        # Define column mapping for DataFrame
        column_mapping = {
            0: f'T2-FLAIR_{self.volume_index}',
            1: f'T1_{self.volume_index}',
            2: f'T1Gd_{self.volume_index}',
            3: f'T2_{self.volume_index}'
        }

        rm_features = pd.DataFrame(features_sets)
        print('get Extracted data')
        result = []
        for ft in RADIOMIC_FEATURES:
           # print(rm_features[ft])
            mean = rm_features.loc[:,ft].mean()
            #print(ft, ' : ',mean)
            result.append(mean)

        return result
    

    def conventional_features(self):
        area = max(self.Maximum_tumout_area())
        diameter = max(self.tumour_diameter())
        involvement = self.glioma_cortex_invasion()
        return area, diameter, involvement