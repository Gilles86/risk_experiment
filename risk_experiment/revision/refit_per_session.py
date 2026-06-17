"""Refit the per-session cognitive models used for the per-session Bayesian
p-values reported in the manuscript (de Hollander et al.).

The original per-session traces that produced the reported 3T/7T values
(`ses-{3t2,7t2}_model-1` for the prior-mean/evidence comparisons and
`ses-{3t2,7t2}_model-neural3` for the neural-regression comparisons) were
either overwritten by later refits or lost. This script regenerates them
cleanly from the current code + data so the reported per-session p-values are
reproducible.

Combined ("both") values are NOT refit here: they come from the existing
combined-session models (model-12 / model-neural32), per the authors.

Output traces go to a dedicated folder so the existing
derivatives/cogmodels/*.netcdf are never clobbered:
    {bids}/derivatives/cogmodels/revision_refit/ses-{session}_model-{label}_trace.netcdf

MUST be run with the analysis-era library stack: the `risk7t` env, whose
editable `bauer` resolves to the pinned 0.1.0 worktree at `src/bauer` (commit
e246d78). risk7t's PyMC (5.25.1) is newer than the 5.10.3 the models were
written against, so we apply `style.shim_pymc_for_bauer(pm)` before building
any model (drops the removed `pm.Data(mutable=...)` kwarg; changes no numbers).

Run (locally):
    ~/mambaforge/envs/risk7t/bin/python risk_experiment/revision/refit_per_session.py
    ~/mambaforge/envs/risk7t/bin/python risk_experiment/revision/refit_per_session.py --smoke
"""
import os
import os.path as op
import argparse
import arviz as az

import sys
sys.path.insert(0, op.join(op.dirname(op.abspath(__file__)), '..', '..'))

# fit_model.py imports RNPRegressionModel, which was removed from the current
# bauer. We don't use it (model-1=RiskModel, neural3=RiskRegressionModel), so
# shim it into bauer.models before importing fit_model to avoid the ImportError.
import bauer.models as _bm
if not hasattr(_bm, 'RNPRegressionModel'):
    _bm.RNPRegressionModel = _bm.RiskRegressionModel

import pymc as pm
from risk_experiment.figures.style import shim_pymc_for_bauer
shim_pymc_for_bauer(pm)  # let pinned bauer 0.1.0 build under PyMC 5.25.1

from risk_experiment.cogmodels.fit_model import get_data, build_model

# Methods: 4 chains, 3000 samples, 1500 burnin
DRAWS = 3000
TUNE = 1500

# (model_label, session) pairs to refit
JOBS = [
    ('1', '3t2'),
    ('1', '7t2'),
    ('neural3', '3t2'),
    ('neural3', '7t2'),
]


def target_accept_for(model_label):
    if model_label.startswith('neural3'):
        return 0.925
    return 0.9


def fit_one(model_label, session, bids_folder, out_folder, draws, tune):
    print(f'\n=== Refitting model-{model_label} ses-{session} '
          f'(draws={draws}, tune={tune}) ===', flush=True)
    df = get_data(model_label, session, bids_folder, roi=None)
    model = build_model(model_label, df, roi=None)
    trace = model.sample(draws=draws, tune=tune,
                         target_accept=target_accept_for(model_label))
    fn = op.join(out_folder, f'ses-{session}_model-{model_label}_trace.netcdf')
    az.to_netcdf(trace, fn)
    print(f'    -> saved {fn}', flush=True)
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    parser.add_argument('--smoke', action='store_true',
                        help='quick 50-draw run to validate the pipeline')
    args = parser.parse_args()

    out_folder = op.join(args.bids_folder, 'derivatives', 'cogmodels', 'revision_refit')
    os.makedirs(out_folder, exist_ok=True)

    draws, tune = (50, 50) if args.smoke else (DRAWS, TUNE)
    jobs = JOBS[:1] if args.smoke else JOBS

    for model_label, session in jobs:
        fit_one(model_label, session, args.bids_folder, out_folder, draws, tune)


if __name__ == '__main__':
    main()
