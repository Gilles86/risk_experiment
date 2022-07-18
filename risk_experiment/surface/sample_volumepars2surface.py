import argparse
import os.path as op
import nipype.pipeline.engine as pe
from fmriprep.workflows.bold import init_bold_surf_wf
import nipype.interfaces.utility as niu
from nipype.interfaces.io import ExportFile
from itertools import product
from nipype.utils.misc import flatten


subject = '02'
session = '7t2'
bids_folder = '/data/ds-risk'

def main(subject, session, bids_folder='/data', smoothed=False):
    parameters = ['r2', 'mu', 'sd']
    spaces = ['fsnative', 'fsaverage']


    key = 'encoding_model'

    if smoothed:
        key += '.smoothed'

    print(key)
    wf = pe.Workflow(name=f'resample_{subject}_{session}_{key.replace(".", "_")}', base_dir='/tmp')


    for par in parameters:
        surf_wf = init_bold_surf_wf(4, spaces, True, name=f'sample_{par}')
        
        inputnode = pe.Node(niu.IdentityInterface(fields=['source_file', 'subjects_dir', 'subject_id', 't1w2fsnative_xfm']),
                       name=f'inputnode_{par}')

        inputnode.inputs.source_file = op.join(bids_folder, f'derivatives/{key}/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_desc-{par}.optim_space-T1w_pars.nii.gz')

        inputnode.inputs.subjects_dir = op.join(bids_folder, 'derivatives', 'freesurfer')
        inputnode.inputs.subject_id = f'sub-{subject}'
        inputnode.inputs.t1w2fsnative_xfm = op.join(bids_folder, f'derivatives/fmriprep/sub-{subject}/anat/sub-{subject}_from-T1w_to-fsnative_mode-image_xfm.txt')

        wf.connect([(inputnode, surf_wf, [('source_file', 'inputnode.source_file'),
                                       ('subjects_dir', 'inputnode.subjects_dir'), 
                                                                ('subject_id', 'inputnode.subject_id'),
                                                                ('t1w2fsnative_xfm', 'inputnode.t1w2fsnative_xfm')])])

        export_file = pe.MapNode(ExportFile(clobber=True), iterfield=['in_file', 'out_file'], 
                                 name=f'exporter_{par}')

        export_file.inputs.out_file = [op.join(bids_folder, f'derivatives/{key}/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_desc-{par}.volume.optim_space-{space}_hemi-{hemi}.func.gii') for space, hemi in product(spaces, ['L', 'R'])]

        wf.connect(surf_wf, ('outputnode.surfaces', flatten), export_file, 'in_file')

    wf.run(plugin='MultiProc', plugin_args={'n_procs' : 4})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    parser.add_argument(
        '--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.session, args.bids_folder,smoothed=args.smoothed )
