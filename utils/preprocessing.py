import SimpleITK as sitk
import numpy as np
import os
import shutil



####################################################################
# Used to resample CTs with varying spacing, direction, origin, and 
# orientation to bring standardization for all the CTs

####################################################################

def get_attributes(sitk_image):
    """Get physical space attributes (meta-data) of the image."""
    attributes = {}
    attributes['orig_pixelid'] = sitk_image.GetPixelIDValue()
    attributes['orig_origin'] = sitk_image.GetOrigin()
    attributes['orig_direction'] = sitk_image.GetDirection()
    attributes['orig_spacing'] = np.array(sitk_image.GetSpacing())
    attributes['orig_size'] = np.array(sitk_image.GetSize(), dtype=int)
    return attributes


def resample_sitk_image(sitk_image,
                        new_spacing=[1, 1, 1],
                        new_size=None,
                        attributes=None,
                        interpolator=sitk.sitkLinear,
                        new_origin=(0.0, 0.0, 0.0),
                        fill_value=0):
    """
    Resample a SimpleITK Image.

    Parameters
    ----------
    sitk_image : sitk.Image
        An input image.
    new_spacing : list of int
        A distance between adjacent voxels in each dimension given in physical units (mm) for the output image.
    new_size : list of int or None
        A number of pixels per dimension of the output image. If None, `new_size` is computed based on the original
        input size, original spacing and new spacing.
    attributes : dict or None
        The desired output image's spatial domain (its meta-data). If None, the original image's meta-data is used.
    interpolator
        Available interpolators:
            - sitk.sitkNearestNeighbor : nearest
            - sitk.sitkLinear : linear
            - sitk.sitkGaussian : gaussian
            - sitk.sitkLabelGaussian : label_gaussian
            - sitk.sitkBSpline : bspline
            - sitk.sitkHammingWindowedSinc : hamming_sinc
            - sitk.sitkCosineWindowedSinc : cosine_windowed_sinc
            - sitk.sitkWelchWindowedSinc : welch_windowed_sinc
            - sitk.sitkLanczosWindowedSinc : lanczos_windowed_sinc
    fill_value : int or float
        A value used for padding, if the output image size is less than `new_size`.

    Returns
    -------
    sitk.Image
        The resampled image.

    Notes
    -----
    This implementation is based on https://github.com/deepmedic/SimpleITK-examples/blob/master/examples/resample_isotropically.py
    """
    sitk_interpolator = interpolator

    # provided attributes:
    if attributes:
        orig_pixelid = attributes['orig_pixelid']
        orig_origin = sitk_image.GetOrigin()
        orig_direction = attributes['orig_direction']
        orig_spacing = attributes['orig_spacing']
        orig_size = attributes['orig_size']

    else:
        # use original attributes:
        orig_pixelid = sitk_image.GetPixelIDValue()
        orig_origin = sitk_image.GetOrigin()
        orig_direction = sitk_image.GetDirection()
        orig_spacing = np.array(sitk_image.GetSpacing())
        orig_size = np.array(sitk_image.GetSize(), dtype=int)

    # new image size:
    if not new_size:
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(int)  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]  # SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(new_size)
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetInterpolator(sitk_interpolator)
    resample_filter.SetOutputOrigin(orig_origin)
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetOutputDirection(orig_direction)
    resample_filter.SetDefaultPixelValue(fill_value)
    resample_filter.SetOutputPixelType(orig_pixelid)

    resampled_sitk_image = resample_filter.Execute(sitk_image)
    return resampled_sitk_image




####################################################################
# Used to resample GTs from the clinicians using CTs 

####################################################################

def buildLabelArrayFromNRRDsegFile(PathNRRDseg, PathNRRDscan):
    '''
    Inputs:
        PathNRRDseg: path of .seg.nrrd file
        PathNRRDscan: path of related CT scan file path

    Output:
        np_LabelScan3D = np array of the same size as scan with unique label for each roi in .seg.nrrd file
    '''

    reader = sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")
    reader.SetFileName(PathNRRDscan)
    SitK_scan3D = reader.Execute()

    np_scan3D = sitk.GetArrayFromImage(SitK_scan3D)
    if len(np_scan3D.shape) == 4:
        np_scan3D = np_scan3D[:,:,:,0]

    reader.SetFileName(PathNRRDseg)
    SitK_seg3D = reader.Execute()

    for k in SitK_seg3D.GetMetaDataKeys():
        v = SitK_seg3D.GetMetaData(k)

    # print(SitK_seg3D)
    np_seg3D = sitk.GetArrayFromImage(SitK_seg3D)
    print(np_seg3D.shape)
    if len(np_seg3D.shape) == 4:
        np_seg3D = np_seg3D[:,:,:,0]

    # below combine all roi's in one np array
    np_seg3D_all = np.zeros(np_seg3D.shape[0:3], dtype=int)
    label_tot = []
    if len(np_seg3D.shape) == 4:
        for iROI in np.arange(np_seg3D.shape[3]):
            segment_str = SitK_seg3D.GetMetaData("Segment" + str(iROI) + "_ID")
            label_int = iROI + 1
            label_roi = segment_str + '_label_' + str(label_int)

            segment_ID = "Segment" + str(iROI) + "_ID"
            segment_Name = "Segment" + str(iROI) + "_Name"
            segment_id = SitK_seg3D.GetMetaData(segment_ID)
            segment_name = SitK_seg3D.GetMetaData(segment_Name)
            label_tot.append(segment_name)

            np_seg3D_all = np_seg3D_all + np_seg3D[:, :, :, iROI] * label_int
    else:
        for iROI in np.arange(1):
            segment_str = SitK_seg3D.GetMetaData("Segment" + str(iROI) + "_ID")
            label_int = iROI + 1
            label_roi = segment_str + '_label_' + str(label_int)

            segment_ID = "Segment" + str(iROI) + "_ID"
            segment_Name = "Segment" + str(iROI) + "_Name"
            segment_id = SitK_seg3D.GetMetaData(segment_ID)
            segment_name = SitK_seg3D.GetMetaData(segment_Name)

            label_tot.append(segment_name)
            np_seg3D_all = np_seg3D_all + np_seg3D[:, :, :] * label_int

    sz_seg3D_all = np_seg3D_all.shape
    offset = SitK_seg3D.GetMetaData("Segmentation_ReferenceImageExtentOffset")
    offset = [int(n) for n in offset.split()]

    np_LabelScan3D = np.zeros(np_scan3D.shape, dtype=int)
    
    np_LabelScan3D[offset[2]:offset[2] + sz_seg3D_all[0], offset[1]:offset[1] + sz_seg3D_all[1],
                   offset[0]:offset[0] + sz_seg3D_all[2]] = np_seg3D_all

    return np_LabelScan3D



####################################################################
# Move files from one directory to another

####################################################################

# # Set the source and destination directories
# src_dir = '/home/ikboljon.sobirov/data/nas/ikboljon.sobirov/rafail/adiporedox_2nd_resampled/'
# dst_dir = '/home/ikboljon.sobirov/data/nas/ikboljon.sobirov/rafail/adiporedox_3nd_resampled/'

# # Set the file extension to move
# file_ext = '_CXaorta.seg.nrrd'

def move_or_copy_files_from_one_dir_to_another(src_dir, dst_dir, file_ext, move=True):
    '''
    Can move or copy files from one dir to another. Similar to scp on terminal.
    Check if you want to move a list of files in a directory, each file with some ext, 
    or if you want to move folders within the src_dir to dst_dir.
    '''

    # get patients 
    patients = os.listdir(src_dir)

    # Get a list of all files in the source directory with the specified extension
    # files_to_move = [f for f in os.listdir(src_dir) if f.endswith(file_ext)]

    # Move each file to the destination directory
    for f in patients:
        src_path = os.path.join(src_dir, f, f+file_ext)
        
        # check if the dstdit/f does not exist, then create it
        if not os.path.exists(os.path.join(dst_dir, f)):
            os.makedirs(os.path.join(dst_dir, f))
            
        dst_path = os.path.join(dst_dir, f, f+file_ext)
        if os.path.exists(src_path):
            if move:
                shutil.move(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)
    