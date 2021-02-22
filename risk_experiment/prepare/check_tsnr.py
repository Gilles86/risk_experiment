import argparse
import os.path as op
from niworkflows.utils.bids import collect_data
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.algorithms.confounds import TSNR
from niworkflows.interfaces.bids import DerivativesDataSink


def main(subject, session, bids_folder, preprocessed=False, cleaned=False):

    root_folder = bids_folder

    bids_filters = {'bold': {'datatype': 'func',
                             'session': session}}

    if preprocessed or cleaned:
        root_folder = op.join(root_folder, 'derivatives', 'fmriprep')
        # bids_filters['bold']['space'] = 'T1w'

    bids_root, _ = collect_data(root_folder, subject,
                                bids_filters=bids_filters,
                                bids_validate=False)

    if preprocssed:
        preproc_suffix = '_preproc'
    elif cleaned:
        preproc_suffix = '_cleaned'
    else:
        preproc_suffix = ''

    workflow = pe.Workflow(name=f'tsnr_sub-{subject}_ses-{session}{preproc_suffix}',
                           base_dir='/tmp')

    inputnode = pe.Node(niu.IdentityInterface(fields=['bold']),
                        name='inputnode')

    inputnode.inputs.bold = bids_root['bold']
    print(bids_root['bold'])

    def clean_data(input_file, fmriprep_regressors, retroicor_regressors):
        from nilearn import image

        fmriprep_confounds_include = ['global_signal', 'dvars', 'framewise_displacement', 'trans_x',
                                      'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                      'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'cosine00', 'cosine01', 'cosine02',
                                      'non_steady_state_outlier00', 'non_steady_state_outlier01', 'non_steady_state_outlier02']

        fmriprep_confounds = pd.read_table(fmriprep_regressors)[fmriprep_confounds_include]
    retroicor_confounds = [
        op.join(sourcedata, f'derivatives/physiotoolbox/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-mapper_run-{run}_desc-retroicor_timeseries.tsv') for run in runs]
    retroicor_confounds = [pd.read_table(
        cf, header=None, usecols=range(18)) for cf in retroicor_confounds]

    confounds = [pd.concat((rcf, fcf), axis=1) for rcf, fcf in zip(
        retroicor_confounds, fmriprep_confounds)]
    confounds = [c.fillna(method='bfill') for c in confounds]

    tsnr = pe.MapNode(TSNR(), iterfield=['in_file'], name='tsnr')

    workflow.connect(inputnode, 'bold', tsnr, 'in_file')

    ds = pe.MapNode(DerivativesDataSink(out_path_base='tsnr',
                                        base_directory=op.join(bids_folder, 'derivatives')),  # , suffix='tsnr'),
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
