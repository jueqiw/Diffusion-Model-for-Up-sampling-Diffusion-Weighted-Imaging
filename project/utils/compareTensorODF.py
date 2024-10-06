###########################################################################
# Necessary packages
###########################################################################

import os
import numpy as np
import concurrent.futures

from dipy.core.gradients import gradient_table

from dipy.data import get_fnames, small_sphere, default_sphere
from dipy.data import get_sphere

from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti

from dipy.reconst.csdeconv import auto_response_ssst
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.reconst.dti import TensorModel
from dipy.reconst.dti import fractional_anisotropy, color_fa

import dipy.reconst.dti as dti

from nilearn import image as img
sphere = get_sphere('repulsion724')

import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

from reformatOutput import *

###########################################################################
# Helper functions
###########################################################################


#---------------------------------------------
# use this for HCP data
#---------------------------------------------

def select_images_by_bvals(diffusion4D_data, bvals, bvecs):

    '''
    For each single subject, selecting 3D images that
    correspond to b=1000

    input: single subject data
        diffusion4D_data.shape: (145, 174, 145, 288)
        bvals.shape: (288,)
        bvecs.shape: (3, 288)

    output: images and bvecs correspond to b=1000
        selected_images.shape: (145, 174, 145, 90)
        selected_bvecs.shape: (3,90)
    '''

    target_bvals=[5,990, 995, 1000, 1005]

    selected_images = []
    selected_bvecs = []
    selected_bvals = []

    for bval in target_bvals:
        indices = np.where(bvals == bval)[0]
        for idx in indices:
            selected_images.append(diffusion4D_data[..., idx])
            selected_bvecs.append(bvecs[idx,:])
            selected_bvals.append(bvals[idx])
            

    selected_images = np.array(selected_images) # shape: (xxx, 145, 174, 145)
    selected_images = np.transpose(selected_images, (1, 2, 3, 0)) # shape: (145, 174, 145, xxx)
    selected_bvecs = np.array(selected_bvecs)
    selected_bvals = np.array(selected_bvals)

    return selected_images, selected_bvecs, selected_bvals

#---------------------------------------------
# use the following two for output data
#---------------------------------------------

def check_integrity(subjid, baseDir):
    # Convert subjid to string in case it is provided as an integer
    subjid_str = str(subjid)
    sourceDir = os.path.join(baseDir, subjid_str)
    
    # List all files in the subject's directory
    all_files = os.listdir(sourceDir)
    
    # Filter files belonging to the given subject
    subject_files = [f for f in all_files if f.startswith("slice_") and f.endswith(".npz")]
    
    # Group files by slice number
    slices = {}
    for f in subject_files:
        slice_num = int(f.split('_')[1])
        if slice_num not in slices:
            slices[slice_num] = []
        slices[slice_num].append(f)
    
    # Check if there are exactly 145 slices
    if len(slices) != 145:
        return False
    
    # Check if each slice has both direction_00 and direction_30 files
    for slice_num in sorted(slices.keys()):
        slice_files = slices[slice_num]
        
        try:
            dir_00_file = next(f for f in slice_files if 'direction_00' in f)
            dir_30_file = next(f for f in slice_files if 'direction_30' in f)
        except StopIteration:
            return False
    
    return True



def combine_output_slices(subjid, sourceDir='/projectnb/ec890/projects/tmp/diffusion_Inverted/', fix_slice_num=False):

    '''
    This function combines the slices of a single subject into a single 4D array.

    Output: matrices of size (60, 145, 174, 145), corresponding to 60 directions, 145 slices each of size 145x174
    '''

    combined_target_slices = []
    combined_predicted_slices = []
    
    subjDir = os.path.join(sourceDir, str(subjid) + '/')
    all_files = os.listdir(subjDir)
    subject_files = [f for f in all_files if f.startswith(f"slice")]
    
    # Group files by slice number
    slices = {}
    for f in subject_files:
        slice_num = int(f.split('_')[1])
        if slice_num not in slices:
            slices[slice_num] = []
        slices[slice_num].append(f)
    # example:
    # slices = {0: ['subjid_slice_0_direction_00.npz', 'subjid_slice_0_direction_30.npz'],
    #           1: ['subjid_slice_1_direction_00.npz', 'subjid_slice_1_direction_30.npz']}
    

    for slice_num in sorted(slices.keys()):
        slice_files = slices[slice_num]
        
        dir_00_file = next(f for f in slice_files if 'direction_00' in f)
        dir_30_file = next(f for f in slice_files if 'direction_30' in f)
        
        dir_00_data = np.load(os.path.join(subjDir, dir_00_file))
        dir_30_data = np.load(os.path.join(subjDir, dir_30_file))
        
        target_00 = dir_00_data['target'].reshape((30, 145, 174))
        target_30 = dir_30_data['target'].reshape((30, 145, 174))
        predicted_00 = dir_00_data['predicted'].reshape((30, 145, 174))
        predicted_30 = dir_30_data['predicted'].reshape((30, 145, 174))
        
        combined_target = np.concatenate((target_00, target_30), axis=0)
        combined_predicted = np.concatenate((predicted_00, predicted_30), axis=0)
        
        combined_target = combined_target[..., np.newaxis]
        combined_predicted = combined_predicted[..., np.newaxis]

        combined_target_slices.append(combined_target)
        combined_predicted_slices.append(combined_predicted)
    
    subjid_target = np.concatenate(combined_target_slices, axis=3)
    subjid_predicted = np.concatenate(combined_predicted_slices, axis=3)

    # Add a slice of zeros to the beginning of the 4D array (slice 000 is not included in the generation process)
    if fix_slice_num:
        subjid_target = np.concatenate((np.zeros((60, 145, 174, 1)), subjid_target), axis=3)
        subjid_predicted = np.concatenate((np.zeros((60, 145, 174, 1)), subjid_predicted), axis=3)
    return subjid_target, subjid_predicted



def append_b0_and_ref(subjid, predicted, target):

    bvals_file = f'/projectnb/connectomedb/HCP1200/{subjid}/T1w/Diffusion/bvals'
    bvecs_file = f'/projectnb/connectomedb/HCP1200/{subjid}/T1w/Diffusion/bvecs'
    nii_file = f'/projectnb/connectomedb/HCP1200/{subjid}/T1w/Diffusion/data.nii.gz'

    data, _, _ = load_nifti(nii_file, return_img=True)
    all_bvals, all_bvecs = read_bvals_bvecs(bvals_file, bvecs_file) # bvals: (288,), bvecs: (288, 3)

    bvecs=[]
    bvals=[]

    ############# ref + [predicted] -> [ref predicted] #############

    # combine the ref slices
    source_dir = '/projectnb/ec890/projects/data/'
    for slice in range(145):
        ref_image_file_path = os.path.join(source_dir, f'{subjid}_slice{slice}_ref.npz')
        ref_image = np.load(ref_image_file_path)['arr_0']
        if slice == 0:
            ref_images = ref_image[..., np.newaxis]
        else:
            ref_images = np.concatenate((ref_images, ref_image[..., np.newaxis]), axis=3) # shape: (30, 145, 174, 145)

    # combine with the output slices
    target = np.concatenate((ref_images, target), axis=0) # shape: (30+60, 145, 174, 145)
    predicted = np.concatenate((ref_images, predicted), axis=0) # shape: (30+60, 145, 174, 145)

    # update the bvecs and bvals
    # for ref:
    bvec_file_path = os.path.join(source_dir, f'{subjid}_bvec{slice}_ref.npz')
    bvecs = np.load(bvec_file_path)['arr_0'].T # shape: (30, 3)
    for slice in range(30):
        bvals.append(1000)
    # for predicted, append to the existing bvecs and bvals
    bvec_file_path = os.path.join(source_dir, f'{subjid}_bvec{slice}_rem.npz')
    bvecs = np.concatenate((bvecs, np.load(bvec_file_path)['arr_0'].T), axis=0) # shape: (30+60, 3)
    for slice in range(60):
        bvals.append(1000)
    bvals = np.array(bvals) # shape: (90,)

    

    ############# b0 + [ref predicted] -> [b0 ref predicted] #############

    indices = np.where(all_bvals == 5)[0] # shape: (18,)
    b0_images = []
    b0_bvecs = []
    b0_bvals = []
    for idx in indices:
        b0_images.append(data[..., idx])
        b0_bvecs.append(all_bvecs[idx,:])
        b0_bvals.append(all_bvals[idx])
    
    b0_images = np.array(b0_images) # shape: (18, 145, 174, 145)
    b0_bvecs = np.array(b0_bvecs) # shape: (18, 3)
    b0_bvals = np.array(b0_bvals) # shape: (18,)

    predicted = np.concatenate((b0_images, predicted), axis=0) # shape: (18+30+60, 145, 174, 145)
    target = np.concatenate((b0_images, target), axis=0) # shape: (18+30+60, 145, 174, 145)

    # update the bvecs and bvals

    bvecs = np.concatenate((b0_bvecs, bvecs), axis=0) # shape: (18+30+60, 3)
    bvals = np.concatenate((b0_bvals, bvals), axis=0) # shape: (18+30+60,)


    ############# final adjustment and saving #############

    target = np.transpose(target, (1, 2, 3, 0)) # shape: (145, 174, 145, 108)
    predicted = np.transpose(predicted, (1, 2, 3, 0)) # shape: (145, 174, 145, 108)

    return {'subjid': subjid, 'target': target, 'predicted': predicted, 'bvals': bvals, 'bvecs': bvecs}


#---------------------------------------------
# compute tensor and fODF
#---------------------------------------------

def compute_tensor_odf(data, bvals, bvecs, computeODF = False):

    '''
    Make sure the data is in the shape of (145, 174, 145, xxx), not (xxx, 145, 174, 145)
    '''
    
    gtab = gradient_table(bvals, bvecs)

    # tensor
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data)
    tensor_vals = dti.lower_triangular(tenfit.quadratic_form)

    FA = tenfit.fa
    FA[np.isnan(FA)] = 0

    MD = tenfit.md
    MD[np.isnan(MD)] = 0

    # fODF
    if computeODF:
        response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=6)
        csd_fit = csd_model.fit(data)
        fodf = csd_fit.odf(small_sphere)
        return {'tensor_vals': tensor_vals, 'FA': FA, 'MD': MD, 'fODF': fodf}
    else:
        return {'tensor_vals': tensor_vals, 'FA': FA, 'MD': MD}


###########################################################################
# A big wrapper function
###########################################################################

def process_single_subject(subjid, 
                           sourceDir='/projectnb/ec890/projects/tmp/', 
                           fix_slice_num=False, 
                           computeODF=False):


    HCPdata, _, _ = load_nifti(f'/projectnb/connectomedb/HCP1200/{subjid}/T1w/Diffusion/data.nii.gz', return_img=True)
    HCPbvals, HCPbvecs = read_bvals_bvecs(f'/projectnb/connectomedb/HCP1200/{subjid}/T1w/Diffusion/bvals', 
                                        f'/projectnb/connectomedb/HCP1200/{subjid}/T1w/Diffusion/bvecs')
    HCPdata, HCPbvecs, HCPbvals = select_images_by_bvals(HCPdata, HCPbvals, HCPbvecs)
    HCP_metrics = compute_tensor_odf(HCPdata, HCPbvals, HCPbvecs, computeODF)


    diffusion_path = os.path.join(sourceDir, 'diffusion_Inverted/')
    target, predicted = combine_output_slices(subjid, diffusion_path, fix_slice_num)
    input_dict = append_b0_and_ref(subjid, predicted, target)
    diff_pred_metrics = compute_tensor_odf(input_dict['predicted'], HCPbvals, input_dict['bvecs'], computeODF)


    GAN_path = os.path.join(sourceDir, 'GAN_Inverted/')
    target, predicted = combine_output_slices(subjid, GAN_path, fix_slice_num)
    input_dict = append_b0_and_ref(subjid, predicted, target)
    GAN_pred_metrics = compute_tensor_odf(input_dict['predicted'], HCPbvals, input_dict['bvecs'], computeODF)
    

    tensorVal_diffusion_diff = np.mean( np.abs(diff_pred_metrics['tensor_vals'] - HCP_metrics['tensor_vals']) )
    tensorVal_GAN_diff = np.mean( np.abs(GAN_pred_metrics['tensor_vals'] - HCP_metrics['tensor_vals']) )

    FA_diffusion_diff = np.mean( np.abs(diff_pred_metrics['FA'] - HCP_metrics['FA']) )
    FA_GAN_diff = np.mean( np.abs(GAN_pred_metrics['FA'] - HCP_metrics['FA']) )

    MD_diffusion_diff = np.mean( np.abs(diff_pred_metrics['MD'] - HCP_metrics['MD']) )
    MD_GAN_diff = np.mean( np.abs(GAN_pred_metrics['MD'] - HCP_metrics['MD']) )

    # print the differences
    #print(f"\nSubject {subjid}")
    #print(f"Tensor value difference (Diffusion): {tensorVal_diffusion_diff}, (GAN): {tensorVal_GAN_diff}")
    #print(f"FA difference (Diffusion): {FA_diffusion_diff}, (GAN): {FA_GAN_diff}")
    #print(f"MD difference (Diffusion): {MD_diffusion_diff}, (GAN): {MD_GAN_diff}")

    return {'subjid': subjid, 
            'diffusion_tensor_diff': tensorVal_diffusion_diff, 
            'GAN_tensor_diff': tensorVal_GAN_diff, 
            'diffusion_FA_diff': FA_diffusion_diff,
            'GAN_FA_diff': FA_GAN_diff,
            'diffusion_MD_diff': MD_diffusion_diff,
            'GAN_MD_diff': MD_GAN_diff}
    
    

    