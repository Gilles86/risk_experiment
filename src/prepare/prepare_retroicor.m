%% Example script using PhysIO with Matlab only (no SPM needed)
%  For documentation of the parameters, see also tapas_physio_new (e.g., via edit tapas_physio_new)

%% Create default parameter structure with all fields
physio = tapas_physio_new();

%% Individual Parameter settings. Modify to your need and remove default settings
physio.save_dir = {'/data2/ds-risk/derivatives/phyiotoolbox'};
physio.log_files.vendor = 'Philips';
physio.log_files.cardiac = {'/data2/ds-risk/sourcedata/sub-05/mri/ses-7t1/SCANPHYSLOG_ri_15102020_1614557_13_1_ses-7t1_task-mapper_run-1V4.log'};
physio.log_files.respiration = {'/data2/ds-risk/sourcedata/sub-05/mri/ses-7t1/SCANPHYSLOG_ri_15102020_1614557_13_1_ses-7t1_task-mapper_run-1V4.log'};
physio.log_files.relative_start_acquisition = 0;
physio.log_files.align_scan = 'last';
physio.scan_timing.sqpar.Nslices = 76/2;
physio.scan_timing.sqpar.TR = 2.3;
physio.scan_timing.sqpar.Ndummies = 0;
physio.scan_timing.sqpar.Nscans = 125;
physio.scan_timing.sqpar.onset_slice = 76/4;
physio.scan_timing.sync.method = 'gradient_log';
physio.scan_timing.sync.grad_direction = 'z';
physio.scan_timing.sync.zero = 1200;
physio.scan_timing.sync.slice = 1800;
physio.scan_timing.sync.volume = 1800;
% physio.scan_timing.sync.vol_spacing = 0.09;
physio.preproc.cardiac.modality = 'ECG';
physio.preproc.cardiac.filter.include = false;
physio.preproc.cardiac.filter.type = 'butter';
physio.preproc.cardiac.filter.passband = [0.3 9];
physio.preproc.cardiac.initial_cpulse_select.method = 'auto_matched';
physio.preproc.cardiac.initial_cpulse_select.max_heart_rate_bpm = 90;
physio.preproc.cardiac.initial_cpulse_select.file = 'initial_cpulse_kRpeakfile.mat';
physio.preproc.cardiac.initial_cpulse_select.min = 0.4;
physio.preproc.cardiac.posthoc_cpulse_select.method = 'off';
physio.preproc.cardiac.posthoc_cpulse_select.percentile = 80;
physio.preproc.cardiac.posthoc_cpulse_select.upper_thresh = 60;
physio.preproc.cardiac.posthoc_cpulse_select.lower_thresh = 60;
physio.model.orthogonalise = 'none';
physio.model.censor_unreliable_recording_intervals = false;
physio.model.output_multiple_regressors = 'sub-05_ses-7t1_task-mapper_run-1_type-confounds_regressors.txt';
physio.model.output_physio = 'sub-05_ses-7t1_task-mapper_run-1_type-confounds_regressors.mat';
physio.model.retroicor.include = true;
physio.model.retroicor.order.c = 3;
physio.model.retroicor.order.r = 4;
physio.model.retroicor.order.cr = 1;
physio.model.rvt.include = false;
physio.model.rvt.delays = 0;
physio.model.hrv.include = false;
physio.model.hrv.delays = 0;
physio.model.noise_rois.include = false;
physio.model.noise_rois.thresholds = 0.9;
physio.model.noise_rois.n_voxel_crop = 0;
physio.model.noise_rois.n_components = 1;
physio.model.noise_rois.force_coregister = 1;
physio.model.movement.include = false;
% physio.model.movement.file_realignment_parameters = {'rp_fMRI.txt'};
% physio.model.movement.order = 6;
% physio.model.movement.censoring_threshold = [3 Inf];
% physio.model.movement.censoring_method = 'MAXVAL';
physio.model.other.include = false;
physio.verbose.level = 2;
physio.verbose.process_log = cell(0, 1);
physio.verbose.fig_handles = zeros(1, 0);
physio.verbose.fig_output_file = 'PhysIO_output.png';
physio.verbose.use_tabs = false;
physio.verbose.show_figs = true;
physio.verbose.save_figs = false;
physio.verbose.close_figs = false;
physio.ons_secs.c_scaling = 1;
physio.ons_secs.r_scaling = 1;

%% Run physiological recording preprocessing and noise modeling
physio = tapas_physio_main_create_regressors(physio);
