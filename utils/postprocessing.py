from skimage.measure import label
import numpy as np
import SimpleITK as sitk

def fill_void_in_axial(mask):

    mask_copy = mask
    filled_mask = np.zeros_like(mask_copy)
    for z in range(mask_copy.shape[0]):
        filled_mask[z,:,:] = ndimage.binary_fill_holes(mask_copy[z,:,:])

    # Replace the original mask with the filled mask
    mask = filled_mask.astype(np.int16)
    return mask

def remove_if_more_than_one_area_exists(mask):
    """
    Connected components analysis for the mask. 
    The mask is 3D so we need to use connectivity=3. 
    It is a single-class one-item segmentation mask.
    It removes all the masks except the largest one.
    """

    # Load the mask
    # mask = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
    # or_mask = mask.copy()

    # Label each connected component
    mask = mask.astype(np.int32)
    labels = label(mask, connectivity=3)

    # Count the number of voxels in each component
    counts = np.bincount(labels.flat)

    # Find the index of the largest component
    largest_idx = np.argmax(counts[1:]) + 1

    # Remove all components except for the largest one
    mask[labels != largest_idx] = 0

    return mask


def fix_aorta_top_bottom_slices(mask, organ_length=68, num_slices=3):
    """
    Only for the aorta project. It crops the top 3 and bottom 3 slices
      where the aorta is found and replaces them with the one slice prior 
      to fix the issue of having a half a shape. Note it uses only 3 slices.
      If you want to fix for more slices, adapt the code for that.
    """
    # get the seg mask array view
    np_arr_view = mask.copy()

    # extract top/bottom dilation from the image
    mask_where = np.where(np_arr_view>0)
    x_max = np.max(mask_where[0])
    x_min = np.min(mask_where[0])
    print("Total length of original aorta", (x_max - x_min), 'with max and min x values:', x_max, x_min)

    # copy the seg mask sitk and apply cast filter to convert it to uint16
    np_arr_copy = np_arr_view
    for i in range(num_slices):
        np_arr_copy[x_max-i,:,:] = np_arr_view[x_max-num_slices,:,:]
        np_arr_copy[x_min+i,:,:] = np_arr_view[x_min+num_slices,:,:]

    # np_arr_copy[x_max,:,:] = np_arr_view[x_max-5,:,:]
    # np_arr_copy[x_max-1,:,:] = np_arr_view[x_max-5,:,:]
    # np_arr_copy[x_max-2,:,:] = np_arr_view[x_max-5,:,:]
    # np_arr_copy[x_max-3,:,:] = np_arr_view[x_max-5,:,:]
    # np_arr_copy[x_max-4,:,:] = np_arr_view[x_max-5,:,:]

    # np_arr_copy[x_min,:,:] = np_arr_view[x_min+5,:,:]
    # np_arr_copy[x_min+1,:,:] = np_arr_view[x_min+5,:,:]
    # np_arr_copy[x_min+2,:,:] = np_arr_view[x_min+5,:,:]
    # np_arr_copy[x_min+3,:,:] = np_arr_view[x_min+5,:,:]
    # np_arr_copy[x_min+4,:,:] = np_arr_view[x_min+5,:,:]

    # check if total length is more than 68mm, then remove the top/bottom slices
    mask_where = np.where(np_arr_copy>0)
    x_max = np.max(mask_where[0])
    x_min = np.min(mask_where[0])

    diff = x_max - x_min
    

    if diff > organ_length:
        # take the 68mm down from the maximum, and zero out anything above max and below min values
        new_x_min = x_max - organ_length
        # print("New max min x values:", x_max, new_x_min)
        np_arr_copy[:new_x_min,:,:] = 0
        np_arr_copy[x_max:,:,:] = 0

        mask_where = np.where(np_arr_copy>0)
        x_max = np.max(mask_where[0])
        x_min = np.min(mask_where[0])
        diff = x_max - x_min
        print("The max_min difference was larger than", organ_length,"\n",
               "Total length is", diff, 'with max and min x values:', x_max, x_min)
    
    elif diff < organ_length:
        # keep adding slices to top and bottom to make it 68mm
        to_add = (organ_length - diff) / 2
        for i in range(int(to_add)):
            if x_max+i < np_arr_copy.shape[0]:
                np_arr_copy[x_max+i,:,:] = np_arr_copy[x_max,:,:]
            if x_min-i > 0:
                np_arr_copy[x_min-i,:,:] = np_arr_copy[x_min,:,:]
        print("The max_min difference was smaller than", organ_length)
                
    mask_where = np.where(np_arr_copy>0)
    x_max = np.max(mask_where[0])
    x_min = np.min(mask_where[0])
    diff = x_max - x_min
    
    # print("Total length is", diff, 'with max and min x values:', x_max, x_min)      
          
    if diff < organ_length:
        to_add = (organ_length - diff) + 1

        for i in range(int(to_add)):
            np_arr_copy[x_min-i,:,:] = np_arr_copy[x_min,:,:]

        print("The max reached the final slice. The max_min difference is still smaller than", organ_length)
    
    mask_where = np.where(np_arr_copy>0)
    x_max = np.max(mask_where[0])
    x_min = np.min(mask_where[0])
    diff = x_max - x_min
    
    print("The final length is", diff, 'with max and min x values:', x_max, x_min)

    return np_arr_copy


def apply_spherical_extension_to_mask(path_to_mask, radius=3, pv=False, path_to_aorta_mask=None):
    '''
    The function takes in a mask and applies dilation in the given radius
    to go around the mask and produce a new mask. E.g., given the mask is 
    the aorta, it uses the radius to generate the aortic wall in 3mm around it.
    
    Inputs:
        path_to_mask: path to the mask file (e.g., aorta or aorta wall)
        radius: radius of the spherical kernel (i.e., 3 voxels in each direction)
        pv: if True, then add aorta and aorta wall together, and then apply dilation
            if True, you need to provide the path to the aorta mask (given that pat_to_mask is aorta wall)
    Output:
        spherical_loop: spherical extension of the mask
    '''

    seg_image = sitk.ReadImage(path_to_mask)

    if pv:
        aorta_mask = sitk.ReadImage(path_to_aorta_mask)
        aorta_mask_arr = sitk.GetArrayFromImage(aorta_mask)
        seg_image_arr = sitk.GetArrayFromImage(seg_image)
        # if either is -1, then make it 1
        aorta_mask_arr[aorta_mask_arr==-1] = 1
        seg_image_arr[seg_image_arr==-1] = 1

        seg_image_arr = seg_image_arr + aorta_mask_arr
        # check if there is any overlap between aorta and aorta wall, if so, keep them as 1
        seg_image_arr[seg_image_arr>1] = 1
        seg_image = sitk.GetImageFromArray(seg_image_arr)

    # get the seg mask array and find values of non-zero voxels
    np_arr_view = sitk.GetArrayViewFromImage(seg_image)
    unique_values = set(np_arr_view[np_arr_view!=0])

    # copy the seg mask sitk and apply cast filter to convert it to uint16
    dilated_image = seg_image
    cast_filter = sitk.CastImageFilter()
    cast_filter.SetOutputPixelType(sitk.sitkUInt16)
    dilated_image = cast_filter.Execute(dilated_image)

    # dilate the image with 3x3x3 kernel (i.e., 3 voxels in each direction)
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(radius)
    dilate_filter.SetKernelType(sitk.sitkBall)
    for label in unique_values:
        dilate_filter.SetForegroundValue(int(label))
        dilated_image = dilate_filter.Execute(dilated_image)

    # extract top/bottom dilation from the image
    mask_where = np.where(np_arr_view!=0)
    # x = np.median(mask_where[0])
    # y = np.median(mask_where[1])
    # z = np.median(mask_where[2])

    x_max = np.max(mask_where[0])
    x_min = np.min(mask_where[0])
    print("Maximum and minimum x values:", x_max, x_min)
    
    # make top/bottom dilation to zero
    dilated_image_arr = sitk.GetArrayFromImage(dilated_image)
    dilated_image_arr[x_max+1:,:,:] = 0
    dilated_image_arr[:x_min,:,:] = 0
    
    # zero out the aorta region for perivascular extraction
    spherical_loop = (np_arr_view - dilated_image_arr)
    
    # check if it contains -1, if so, make it 1
    spherical_loop[spherical_loop==-1] = 1


    return spherical_loop
