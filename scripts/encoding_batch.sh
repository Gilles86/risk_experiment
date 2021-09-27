for sub in {2..23}; do printf -v sub "%02d" $sub; python /risk_experiment/run_batch.py sub-$sub encoding_model.fit_mapper_surf; done;
for sub in {25..32}; do printf -v sub "%02d" $sub; python /risk_experiment/run_batch.py sub-$sub encoding_model.fit_mapper_surf; done;
