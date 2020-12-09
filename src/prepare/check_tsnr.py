import argparse
import os.path as op
from niworkflows.utils.bids import collect_data
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.algorithms.confounds import TSNR
from niworkflows.interfaces.bids import DerivativesDataSink

def main(subject, session, bids_folder, preprocessed=False):


    root_folder = bids_folder

    bids_filters = {'bold':{'datatype':'func',
                    'session':session}}

    if preprocessed:
        root_folder = op.join(root_folder, 'derivatives', 'fmriprep')
        # bids_filters['bold']['space'] = 'T1w'

    bids_root, _ = collect_data(root_folder, subject,
            bids_filters=bids_filters,
            bids_validate=False)

    preproc_suffix = '_preproc' if preprocessed else ''
    workflow = pe.Workflow(name=f'tsnr_sub-{subject}_ses-{session}{preproc_suffix}',
            base_dir='/tmp')

    inputnode = pe.Node(niu.IdentityInterface(fields=['bold']),
            name='inputnode')

    inputnode.inputs.bold = bids_root['bold']
    print(bids_root['bold'])

    tsnr = pe.MapNode(TSNR(), iterfield=['in_file'], name='tsnr')

    workflow.connect(inputnode, 'bold', tsnr, 'in_file')

    ds = pe.MapNode(DerivativesDataSink(out_path_base='tsnr',
        base_directory=op.join(bids_folder, 'derivatives')),#, suffix='tsnr'),
        iterfield=['in_file', 'source_file'], name='datasink')

    ds.inputs.desc = 'preproc' if preprocessed else 'raw'

    workflow.connect(tsnr, 'tsnr_file', ds, 'in_file')
    workflow.connect(inputnode, 'bold', ds, 'source_file')

    workflow.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    parser.add_argument('--preprocessed', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.session, args.bids_folder, args.preprocessed)
