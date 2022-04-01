import nipype.pipeline.engine as pe
import numpy as np
import os.path as op
import glob
import argparse
from bids import BIDSLayout
import nipype.interfaces.utility as niu
from nipype.interfaces import fsl
from niworkflows.interfaces.bids import DerivativesDataSink
from niworkflows.interfaces.fixes import (FixN4BiasFieldCorrection as N4BiasFieldCorrection,
                                          FixHeaderRegistration as Registration)
# from niworkflows.interfaces.registration import fmap2ref_reg
from niworkflows.interfaces.utils import GenerateSamplingReference
from niworkflows.interfaces.registration import ANTSApplyTransformsRPT as ApplyTransforms
import logging


def main(subject, session, bids_folder, modalities=None, registration_scheme='linear_precise'):

    if modalities is None:
        modalities = ['T2starw', 'MTw', 'TSE']

    curdir = op.dirname(op.realpath(__file__))
    registration_scheme = op.join(curdir, f'{registration_scheme}.json')

    anat_dir = op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'anat')

    target = op.join(bids_folder, 'derivatives', 'fmriprep',
                     f'sub-{subject}', 'anat', f'sub-{subject}_desc-preproc_T1w.nii.gz')
    target_mask = op.join(bids_folder, 'derivatives', 'fmriprep',
                          f'sub-{subject}', 'anat', f'sub-{subject}_desc-brain_mask.nii.gz')

    init_regs = glob.glob(op.join(bids_folder, 'derivatives', 'fmriprep',
                                  f'sub-{subject}', f'ses-{session}', 'anat', '*from-orig_to-T1w_*.txt'))

    t1w_to_mni_transform = op.join(bids_folder, 'derivatives', 'fmriprep',
                                   f'sub-{subject}', 'anat', f'sub-{subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5')

    t1w_in_mni = op.join(bids_folder, 'derivatives', 'fmriprep',
                         f'sub-{subject}', 'anat', f'sub-{subject}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz')

    mni_brain_mask = op.join(bids_folder, 'derivatives', 'fmriprep',
                             f'sub-{subject}', 'anat', f'sub-{subject}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')

    if len(init_regs) > 0:
        init_reg = init_regs[0]
    else:
        init_reg = None

    print(f'INITIAL TRANSFORM: {init_reg}')

    def make_registration_wf(input_file, name, subject=subject, target=target,
                             target_mask=target_mask, init_reg=init_reg, t1w_to_mni_transform=t1w_to_mni_transform,
                             t1w_in_mni=t1w_in_mni,
                             mni_brain_mask=mni_brain_mask,
                             ants_numthreads=8):

        workflow = pe.Workflow(base_dir='/tmp/workflow_folders',
                               name=name)

        input_node = pe.Node(niu.IdentityInterface(fields=['input_file', 'target', 'target_mask', 't1w_to_mni_transform',
                                                           't1w_in_mni',
                                                           'mni_brain_mask']),
                             name='inputspec')
        input_node.inputs.input_file = input_file
        input_node.inputs.target = target
        input_node.inputs.target_mask = target_mask
        input_node.inputs.init_reg = init_reg
        input_node.inputs.t1w_to_mni_transform = t1w_to_mni_transform
        input_node.inputs.t1w_in_mni = t1w_in_mni
        input_node.inputs.mni_brain_mask = mni_brain_mask

        convert_dtype = pe.Node(fsl.maths.MathsCommand(), name='convert_dtype')
        convert_dtype.inputs.output_datatype = 'double'

        workflow.connect(input_node, 'input_file', convert_dtype, 'in_file')

        inu_n4 = pe.Node(
            N4BiasFieldCorrection(
                dimension=3,
                save_bias=True,
                num_threads=ants_numthreads,
                rescale_intensities=True,
                copy_header=True,
            ),
            n_procs=ants_numthreads,
            name="inu_n4",)

        workflow.connect(convert_dtype, 'out_file', inu_n4, 'input_image')

        register = pe.Node(Registration(from_file=registration_scheme, num_threads=ants_numthreads, verbose=True),
                           name='registration')

        workflow.connect(inu_n4, 'output_image', register, 'moving_image')

        if init_reg:
            workflow.connect(input_node, 'init_reg', register,
                             'initial_moving_transform')

        workflow.connect(input_node, 'target', register, 'fixed_image')
        workflow.connect(input_node, 'target_mask',
                         register, 'fixed_image_masks')

        def get_mask(input_image):
            from nilearn import image
            from nipype.utils.filemanip import split_filename
            import os.path as op

            _, fn, _ = split_filename(input_image)
            mask = image.math_img('im != 0', im=input_image)
            new_fn = op.abspath(fn + '_mask.nii.gz')
            mask.to_filename(new_fn)

            return new_fn

        mask_node = pe.Node(niu.Function(function=get_mask,
                                         input_names=['input_image'], output_names=['mask']),
                            name='mask_node')

        workflow.connect(register, 'warped_image', mask_node, 'input_image')

        gen_grid_node = pe.Node(GenerateSamplingReference(),
                                name='gen_grid_node')

        workflow.connect(mask_node, 'mask', gen_grid_node, 'fov_mask')
        workflow.connect(inu_n4, 'output_image',
                         gen_grid_node, 'moving_image')
        workflow.connect(input_node, 'target', gen_grid_node, 'fixed_image')

        datasink_image_t1w = pe.Node(DerivativesDataSink(out_path_base='registration',
                                                         compress=True,
                                                         base_directory=op.join(bids_folder, 'derivatives')),
                                     name='datasink_image_t1w')
        workflow.connect(input_node, 'input_file',
                         datasink_image_t1w, 'source_file')
        datasink_image_t1w.inputs.space = 'T1w'
        datasink_image_t1w.inputs.desc = 'registered'

        datasink_report_t1w = pe.Node(DerivativesDataSink(
            out_path_base='registration',
            space='T1w',
            base_directory=op.join(bids_folder, 'derivatives'),
            datatype='figures'),
            name='datasink_report_t1w')

        workflow.connect(input_node, 'input_file',
                         datasink_report_t1w, 'source_file')
        datasink_report_t1w.inputs.space = 'T1w'

        transformer = pe.Node(ApplyTransforms(interpolation='LanczosWindowedSinc', generate_report=True, num_threads=ants_numthreads),
                              n_procs=ants_numthreads,
                              name='transformer')
        workflow.connect(transformer, 'output_image',
                         datasink_image_t1w, 'in_file')
        workflow.connect(transformer, 'out_report',
                         datasink_report_t1w, 'in_file')
        workflow.connect(inu_n4, 'output_image', transformer, 'input_image')
        workflow.connect(gen_grid_node, 'out_file',
                         transformer, 'reference_image')
        workflow.connect(register, 'composite_transform',
                         transformer, 'transforms')

        concat_transforms = pe.Node(niu.Merge(2), name='concat_transforms')

        workflow.connect(register, 'composite_transform',
                         concat_transforms, 'in2')
        workflow.connect(input_node, 't1w_to_mni_transform',
                         concat_transforms, 'in1')

        transformer_to_mni1 = pe.Node(ApplyTransforms(interpolation='LanczosWindowedSinc', generate_report=False, num_threads=ants_numthreads),
                                      n_procs=ants_numthreads,
                                      name='transformer_to_mni1')
        workflow.connect(inu_n4, 'output_image',
                         transformer_to_mni1, 'input_image')
        workflow.connect(input_node, 't1w_in_mni',
                         transformer_to_mni1, 'reference_image')
        workflow.connect(concat_transforms, 'out',
                         transformer_to_mni1, 'transforms')

        mask_node_mni = pe.Node(niu.Function(function=get_mask,
                                             input_names=['input_image'], output_names=['mask']),
                                name='mask_node_mni')
        workflow.connect(transformer_to_mni1, 'output_image',
                         mask_node_mni, 'input_image')

        def join_masks(mask1, mask2):
            from nilearn import image
            from nipype.utils.filemanip import split_filename
            import os.path as op

            _, fn, _ = split_filename(mask1)

            new_mask = image.math_img(
                '(im1 > 0) & (im2 > 0)', im1=mask1, im2=mask2)

            new_fn = op.abspath(fn + '_jointmask' + '.nii.gz')

            new_mask.to_filename(new_fn)

            return new_fn

        combine_masks_node = pe.Node(niu.Function(function=join_masks,
                                                  input_names=[
                                                      'mask1', 'mask2'],
                                                  output_names=['combined_mask']), name='combine_mask_node')

        workflow.connect(mask_node_mni, 'mask', combine_masks_node, 'mask1')
        workflow.connect(input_node, 'mni_brain_mask',
                         combine_masks_node, 'mask2')

        gen_grid_node_mni = pe.Node(GenerateSamplingReference(),
                                    name='gen_grid_node_mni')
        workflow.connect(combine_masks_node, 'combined_mask',
                         gen_grid_node_mni, 'fov_mask')
        workflow.connect(inu_n4, 'output_image',
                         gen_grid_node_mni, 'moving_image')
        workflow.connect(input_node, 't1w_in_mni',
                         gen_grid_node_mni, 'fixed_image')

        transformer_to_mni2 = pe.Node(ApplyTransforms(interpolation='LanczosWindowedSinc', generate_report=False, num_threads=ants_numthreads),
                                      n_procs=ants_numthreads,
                                      name='transformer_to_mni2')
        workflow.connect(inu_n4, 'output_image',
                         transformer_to_mni2, 'input_image')
        workflow.connect(gen_grid_node_mni, 'out_file',
                         transformer_to_mni2, 'reference_image')
        workflow.connect(concat_transforms, 'out',
                         transformer_to_mni2, 'transforms')

        datasink_image_mni = pe.Node(DerivativesDataSink(out_path_base='registration',
                                                         compress=True,
                                                         base_directory=op.join(bids_folder, 'derivatives')),
                                     name='datasink_mni')
        datasink_image_mni.inputs.source_file = input_file
        datasink_image_mni.inputs.space = 'MNI152NLin2009cAsym'
        datasink_image_mni.inputs.desc = 'registered'

        workflow.connect(input_node, 'input_file',
                         datasink_image_mni, 'source_file')
        workflow.connect(transformer_to_mni2, 'output_image',
                         datasink_image_mni, 'in_file')

        return workflow

    df = BIDSLayout(anat_dir, validate=False).to_df()
    print(df['extension'])
    df = df[np.in1d(df.extension, ['.nii', '.nii.gz'])]

    if 'acquisition' in df.columns:
        df = df[~((df.suffix == 'T2starw') & (df.acquisition != 'average'))]

    print(df)
    df = df[np.in1d(df['suffix'], modalities)]

    for ix, row in df.iterrows():
        logging.info('Registering {row.path}')
        wf_name = f'register_{subject}_{session}_{row.suffix}'

        if ('run' in row) and row.run:
            wf_name += f'_{row.run}'

        wf = make_registration_wf(row.path, wf_name)

        wf.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    parser.add_argument(
        '--registration_scheme', default='linear_precise')
    parser.add_argument(
        '--modalities', default=['T2starw', 'MTw', 'TSE'], nargs="+")
    args = parser.parse_args()

    main(args.subject, args.session, args.bids_folder,
         args.modalities, args.registration_scheme)
