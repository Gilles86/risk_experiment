import argparse
from fmriprep.workflows.bold import init_bold_surf_wf
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import os.path as op
from nipype.interfaces.io import ExportFile
from itertools import product
from nipype.utils.misc import flatten
from nipype.interfaces import fsl

def main(subject, session, bids_folder):

    wf = pe.Workflow(name=f'sample_com_to_fsnative_{subject}_{session}', base_dir='/tmp')

    spaces = ['fsnative', 'fsaverage']

    tms_dir = op.join(bids_folder, 'derivatives', 'tms_targets',
                      f'sub-{subject}', f'ses-{session}', 'func')

    surf_wf = init_bold_surf_wf(4, spaces, True, name=f'sample2surf')

    inputnode = pe.Node(niu.IdentityInterface(fields=['source_file', 'subjects_dir', 'subject_id', 't1w2fsnative_xfm']),
                        name=f'inputnode')

    source_file = op.join(
        tms_dir, f'sub-{subject}_desc-r2smoothnpc_mask.nii.gz')
    print(source_file)
    inputnode.inputs.source_file = source_file

    to_float = pe.Node(fsl.ChangeDataType(output_datatype='float'),
            name='to_float')

    inputnode.inputs.subjects_dir = op.join(
        bids_folder, 'derivatives', 'freesurfer')
    inputnode.inputs.subject_id = f'sub-{subject}'
    inputnode.inputs.t1w2fsnative_xfm = op.join(
        bids_folder, f'derivatives/fmriprep/sub-{subject}/anat/sub-{subject}_from-T1w_to-fsnative_mode-image_xfm.txt')

    wf.connect(inputnode, 'source_file', to_float, 'in_file')
    wf.connect(to_float, 'out_file', surf_wf, 'inputnode.source_file')

    wf.connect([(inputnode, surf_wf, [('subjects_dir', 'inputnode.subjects_dir'),
                                      ('subject_id', 'inputnode.subject_id'),
                                      ('t1w2fsnative_xfm', 'inputnode.t1w2fsnative_xfm')])])

    export_file = pe.MapNode(ExportFile(clobber=True), iterfield=[
                             'in_file', 'out_file'], name=f'exporter')

    export_file.inputs.out_file = [op.join(
        bids_folder, f'derivatives/tms_targets/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_desc-r2com.volume.optim_space-{space}_hemi-{hemi}.func.gii') for space, hemi in product(spaces, ['L', 'R'])]

    wf.connect(surf_wf, ('outputnode.surfaces', flatten),
               export_file, 'in_file')

    wf.run(plugin='MultiProc', plugin_args={'n_procs': 4})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.subject, args.session, args.bids_folder,)
