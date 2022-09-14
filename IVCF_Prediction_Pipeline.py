#===================================================================================#
#
#    Implementation of an IVC 2D Prediction model 
#   The program reads CT images, performs bi-linear interpolation and standardization,
#
# Author/Compiler: Joe Wildenberg, Rahul Gomes, Connor Kamrowski, Cameron Senor, Avi Mohan
# Contribution: Cameron Senor, Avi Mohan, and Joe Wildenberg wrote the code to read in scans then perform bi-linear interpolation and standardize the CT images.
# Rahul Gomes and Connor Kamrowski compiled the code for prediction, overall compilation, and flagIVCfilters
#===================================================================================#

import numpy as np
import sys
import glob
import pydicom as dicom
import os # for length of folder
from scipy import ndimage 
from tensorflow.keras.models import load_model 
import matplotlib.pyplot as plt
from time import time
import more_itertools as mit

#Flags for verbose output
BASIC_OUT = True
VERBOSE_OUT = True
REPORT_OUT = True
PAPER_OUT = True
REPORT_VERBOSE = True
PICS_OUT = True

# None value for global
slice_Thickness = None
axial_Resolution = None
listDir = []

#Paths
processing_path = '../../Database/Extracted_Scans/'

output_path = 'Output/Hard_Normalize_300_7/'
report_file = 'report.txt'
paper_file = 'paper.txt'
total_files = 0
num_sequences = 0

#Default values to crop
SCAN_LIMIT_Z = 400 #40 cm
SCAN_LIMIT_XY = 0.2 #20% removed from each side of slice

#Default dimensions to resize scan
IMAGE_SLICES = 128
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

#Default parameters to consider scan positive for filter
SIGNIFICANT_COUNT = 300 #if number of pixels greater than 200, then flag the scan 
SEQUENCE = 7 #if 7 continous slices have pixels over 200, then flag and print those scans
MIN_HU = 0 #This and all HU below will become 0
MAX_HU = 2500 #This and all HU above will become 1


MODEL = '../Segmentation_Model/Output/Hard_Normalize2/UNet_2_M64.h5'

# Function to read in a scan. Returns 3D numpy array in Hounsfield Units (HU)
def readScan(fileDir):
    # Read in all of the slices
    slices = [dicom.dcmread(s)
        for s in glob.glob(fileDir + '/*.dcm')]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    
    #Get the slice thickness and pixel spacing
    try:
        slice_thickness = np.abs(slices[0]['ImagePositionPatient'].value - 
                                 slices[1]['ImagePositionPatient'].value)
    except:
        slice_thickness = np.abs(slices[0]['SliceLocation'].value - 
                                 slices[1]['SliceLocation'].value)

    pixel_spacing = list(slices[0].PixelSpacing)

    # Setting global variables for slice thickness and axiels resolution.    
    global slice_Thickness
    slice_Thickness = slice_thickness

    global axial_Resolution
    axial_Resolution = pixel_spacing    
        
    
    # Get the slope and intercept
    intercept = slices[0]['RescaleIntercept'].value
    slope = slices[0]['RescaleSlope'].value

    # Make the image a numpy array
    scan = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    
    #Convert to Hounsfield units (HU)
    if slope != 1:
        scan = (slope * scan.astype(np.float64)).astype(np.int16)

    scan += np.int16(intercept)
    print() if BASIC_OUT else None  
    print('Voxels: ' + str(pixel_spacing[0]) + ' x ' + str(pixel_spacing[1]) + 
      ' x ' + str(slice_thickness) + ' mm') if BASIC_OUT else None
    print ('Matrix: ' + str(scan.shape)) if BASIC_OUT else None
        
    return scan


#Remove edges and lower portion of scan as per defined variables above
def cropScan(scan):  
    col_size = scan.shape[1]
    row_size = scan.shape[2]
    
    #Number of slices to keep in z direction
    z_crop = int(SCAN_LIMIT_Z / float(slice_Thickness))
    
    newScan = scan[:z_crop,
                   int(SCAN_LIMIT_XY*col_size):int((1-SCAN_LIMIT_XY)*col_size),
                   int(SCAN_LIMIT_XY*row_size):int((1-SCAN_LIMIT_XY)*row_size)]
    return newScan


#Set the upper and lower HU values, then normalize to [0,1]
def normalize(volume):
    print('Image range before normalize: [' + str(np.amin(volume)) + 
          ',' + str(np.amax(volume)) + ']') if BASIC_OUT else None
    volume[volume < MIN_HU] = MIN_HU
    volume[volume > MAX_HU] = MAX_HU
    volume = (volume - MIN_HU) / (MAX_HU - MIN_HU)
    array_sum_NaN = np.sum(volume)
    if np.isnan(array_sum_NaN):
        volume = np.nan_to_num(volume)
        print('NaN in image!!')

    return volume.astype('float32')

# Resize the scan to the desired dimensions using spline interpolation
def resize_volume(img, slices, width, height):  
    img = np.moveaxis(img, 0, -1)
    # Set the desired depth
    desired_depth = slices
    desired_width = width
    desired_height = height
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    
    # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

# Reads in a scan and does initial processing using the above helper methods
def getScan(scanFolder, slices, width, height):

    # turning folder of DICOM into numpy
    scan = readScan(processing_path + '/' + scanFolder)
   
    #Crop the scan to just the patient and segment out the CT table
    segCropped = cropScan(scan)
    print('Scan size after cropping: ' + 
          str(segCropped.shape)) if VERBOSE_OUT else None
    
    # #normalizing array
    normalized = normalize(segCropped)
    finalScan = resize_volume(normalized, slices, width, height)
    print('Final scan size: ' + 
          str(finalScan.shape)) if VERBOSE_OUT else None

    return np.array([finalScan])
  
    
def displayImage(images_IVC, labels_IVC, sequence_num, img_folder):
    if not os.path.exists(output_path+img_folder):
        os.makedirs(output_path+img_folder)

    x = len(images_IVC)
    plt.figure(figsize=(12, 6*x)) #Each figure is a 6 * 6. So setting length based on the number of scans being printed
    demo_counter = 0
    for i in range(x):
        plt.subplot(x, 2, demo_counter+1)
        demo_counter+=1
        plt.imshow(images_IVC[i], cmap='gray', vmin=0, vmax=0.33)
        plt.axis('off')
        plt.title("Scan_" + str(sequence_num[i]))
        plt.subplot(x, 2, demo_counter+1)
        demo_counter+=1
        plt.imshow(labels_IVC[i],cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title("Segmentation_"+str(sequence_num[i]))
    plt.show()
    plt.savefig(output_path+img_folder+'/IVC_Filter'+str(sequence_num)+'.png', 
                bbox_inches='tight') if PICS_OUT else None
    plt.clf()
   
def flagIvcFilters(images, labels, SC, sequence, location):
    print('#========================================#') if VERBOSE_OUT else None
    found = 'No'
    
    for i in range(len(images)):
        #We are looking for SC number of sequential slices with value>SC for ones
        label = labels[i:i+sequence]
        print(label.shape) if VERBOSE_OUT else None
        
        #the sequence of images are sequence, width, height i.e. 5, 256, 256
        #returns the sum along the length with shape 2, 5
        numPositive = np.sum(label, axis = 1)
        #returns the sum with for each slice with shape 2
        numPositive = np.sum(numPositive, axis = 1)
        print('Num positives are: ' + str(numPositive)) if VERBOSE_OUT else None 

        #If the continuos slices all have volumes beyond the threshold SC 
        #IVC filter considered found
        if (np.all(numPositive >= SC)):
            disp_image = []
            disp_mask = []
            disp_sequence = []
            for j in range(len(label)):
                disp_mask.append(labels[j+i])
                disp_image.append(images[j+i])
                disp_sequence.append(j+i)
                print('Scan: ' + str(j+i) + ' with # Voxels: ' + 
                      str(numPositive)) if VERBOSE_OUT else None
                report.write(str(numPositive) + ' positive pixels predicted in slice ' 
                             + str(j+i) + '\n') if REPORT_VERBOSE else None
            displayImage(disp_image, disp_mask, disp_sequence, location) if PICS_OUT else None
            found = 'Yes'
            
    #Save the final determination of the scan
    report.write(found + '\n') if REPORT_OUT else None
    
 
#This code finds out the longest sequences and prints them instead of repetitive printing of same sequences
def flagIvcFilters_2(images, labels, SC, sequence, location):
    global num_sequences
    print('#========================================#') if VERBOSE_OUT else None
    found = 'No'
    local_sequences = 0
    
    #Find the sum of all IVCF pixels per slice
    #the sequence of images are sequence, width, height i.e. 5, 256, 256
    #returns the sum along the length with shape 2, 5
    numPositive = np.sum(labels, axis = 1)
    #returns the sum with for each slice with shape 2
    numPositive = np.sum(numPositive, axis = 1)
    #Return the index number of the slices that have IVCF greater than a cutoff threshold SC
    ivc_indexes = [i for i,v in enumerate(numPositive) if v >= SC]
    print("Slices with values greater than SC are: {}".format(ivc_indexes)) if VERBOSE_OUT else None 
    
    groups = []
    #Create consecutive groups of slices from ivc_indexes. 
    #Basically if ivc_indexes looks like this: [2, 3, 4, 7, 8, 9, 11, 13, 14]
    #The code below will make it like this: [[2, 3, 4], [7, 8, 9], [11], [13, 14]]
    #this means there are 4 consecutive sequences
    for group in mit.consecutive_groups(ivc_indexes):
        groups.append(list(group))


    #This code finds the sequences that have more than the set sequence_num and sends it out for display
    for i in range(len(groups)):
        # ivcf_values.append([numPositive[j] for j in groups[i]])
        curr_len = len(groups[i])
        if curr_len >= sequence: 
            num_sequences+=1
            local_sequences+=1
            start = (groups[i][0])
            stop = (groups[i][-1])
            displayImage(images[start:stop], labels[start:stop], 
                         groups[i], location) if PICS_OUT else None
            found = 'Yes'
            print("Groups of slices with values greater than SC are: {}".format(groups[i])) if VERBOSE_OUT else None 
            report.write("\nGroups of slices with values greater than SC are: {} \n".format(groups[i])) if REPORT_VERBOSE else None 
    paper.write(location + '\t'+ str(local_sequences) + '\n') if PAPER_OUT else None
    #Save the final determination of the scan
    report.write(found + '\n') if REPORT_OUT else None
    

#===================================================================================#
# Run Prediction
def processScan(scan_path):  
    print('#========================================#') if BASIC_OUT else None
    print('Processing ' + scan_path) if BASIC_OUT else None
    report.write(scan_path + ' ') if REPORT_OUT else None
    
    #initialize the training data
    read_time_1 = time()

    filter_scans = getScan(scan_path, IMAGE_SLICES, IMAGE_WIDTH, IMAGE_HEIGHT)

    print() if BASIC_OUT else None
    print('Time to read images is :' + str(time()-read_time_1)) if BASIC_OUT else None
    print() if BASIC_OUT else None


    #===================================================================================#
    # make sure that every testing image is within the right range (0 to 1)
    for scan in filter_scans:
        # if the maximum or minimum are outside of the expected range, then print the values
        if(np.max(scan) > 1.00001 or np.min(scan) < 0):
            print('Minimum pixel value: ' + str(np.min(scan))) if VERBOSE_OUT else None
            print('Maximum pixel value: ' + str(np.max(scan))) if VERBOSE_OUT else None

    print('#========================================#') if BASIC_OUT else None
    
    ######################################################
    #####Start the TensorFlow pipeline####################
    ######################################################
    
    ###############################
    #The current dimension looks something like 1, 256, 256, 128
    #Remove the first dimension to give 256, 256, 128
    filter_scans = np.squeeze(filter_scans, axis = 0)
    #Transpose dimension to make it 128, 256, 256
    filter_scans = np.transpose(filter_scans, (2, 0, 1))
    #Finally add a single dimension for a grayscale image which is 128, 256, 256, 1
    filter_scans = np.expand_dims(filter_scans, axis = -1)

    #Load a model
    UNet_Prediction = load_model(MODEL, compile = False)
    print(UNet_Prediction.summary()) if VERBOSE_OUT else None
    
      #Predict output
  
    pred_mask_valid = []
    for i in range(len(filter_scans)):
        temp_file = np.expand_dims(filter_scans[i], axis = 0)
        temp_file = UNet_Prediction.predict(temp_file)
        temp_file = np.squeeze(temp_file, axis = 0)
        pred_mask_valid.append(temp_file)
  
       
    print('Prediction Comnpleted!') if VERBOSE_OUT else None
  
    pred_mask_valid = np.asarray(pred_mask_valid)
    print('The shape of prediction is: ' + 
          str(pred_mask_valid.shape)) if VERBOSE_OUT else None
    preds_Unet_valid = pred_mask_valid.argmax(axis=-1)
    print('The shape of prediction after argmax is: ' +
          str(preds_Unet_valid.shape)) if VERBOSE_OUT else None
    pred_demo_valid = preds_Unet_valid.flatten()
    print('The shape of the flattened NumPy array is: ' +
          str(pred_demo_valid.shape)) if VERBOSE_OUT else None
    print('The unique values in predictions is: ' +
          str(np.unique(pred_demo_valid))) if VERBOSE_OUT else None
    #for filter_scans lose the last dimension(128,256,256)
    filter_scans = np.squeeze(filter_scans, axis = -1)
    #flagIvcFilters(filter_scans, preds_Unet_valid, SIGNIFICANT_COUNT, SEQUENCE, scan_path) ###RG_Comment
    flagIvcFilters_2(filter_scans, preds_Unet_valid, SIGNIFICANT_COUNT, SEQUENCE, scan_path) ###RG_CHANGE (Added scan_path)


    
########################################################################
def main(argv):
    start_time = time()
    global processing_path
    global report 
    global paper
    global total_files
    global num_sequences
    report = open(output_path + '/' + report_file, 'w')
    paper = open(output_path + '/' + paper_file, 'w')
    
    paper.write('IVCF Scans' + '\t'+ 'Sequences' + '\n') if PAPER_OUT else None
    #Get list of all scans to process
    listDir = [name for name in os.listdir(processing_path)
            if os.path.isdir(os.path.join(processing_path, name))]
    
    for i in listDir:
        processScan(i)
        total_files+=1
    
    print('Time to process scans :' + str(time()-start_time)) if BASIC_OUT else None
    print('IVCF scans procesed are :' + str(total_files)) if BASIC_OUT else None
    print('Sequences detected are :' + str(num_sequences)) if BASIC_OUT else None
    
    
    
    
    print('#========================================#') if VERBOSE_OUT else None
    print('#========================================#') if VERBOSE_OUT else None
    print('#========================================#') if VERBOSE_OUT else None
    print('#========================================#') if VERBOSE_OUT else None
    print('#========================================#') if VERBOSE_OUT else None
    
    paper.write('Normal Scans' + '\t'+ 'Sequences' + '\n') if PAPER_OUT else None
    
    
    processing_path = '../../Database/Mayo_NoFilter/'
    start_time = time()
    #Get list of all normal scans to process
    listDir_2 = [name for name in os.listdir(processing_path)
            if os.path.isdir(os.path.join(processing_path, name))]
    
    
    #reset_counters
    total_files = 0
    num_sequences = 0
    
    for i in listDir_2:
        processScan(i)
        total_files+=1
    
    print('Time to process normal scans :' + str(time()-start_time)) if BASIC_OUT else None
    print('Normal scans procesed are :' + str(total_files)) if BASIC_OUT else None
    print('Sequences detected are :' + str(num_sequences)) if BASIC_OUT else None
    
    
    report.close()

# runs the main method.
if __name__ == '__main__':

    # call line with any arguments
    main(sys.argv[1:])
    print('done') if BASIC_OUT else None
########%%%%%%%%%%%