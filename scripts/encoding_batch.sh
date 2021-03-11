for sub in {12..23}; do printf -v sub "%02d" $sub; python /risk_experiment/run_batch.py sub-$sub encoding_model.fit_mapper_surf; done;
