from utils import run_experiment
from task import TaskSessionMRI

if __name__ == '__main__':

    session_cls = TaskSessionMRI
    task = 'task'

    for run in range(1, 5):
        run_experiment(session_cls, run=run, session='7t', task=task, settings='7t', use_runs=True)
