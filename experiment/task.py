from exptools2.core import Session
from exptools2.core import Trial
from gamble import IntroBlockTrial, GambleTrial, GambleInstructionTrial
from utils import run_experiment, create_stimulus_array_log_df
from psychopy import logging
from session import PileSession
import numpy as np
from trial import InstructionTrial, DummyWaiterTrial
import os
import os.path as op
import pandas as pd

class TaskSession(PileSession):

    Trial = GambleTrial

    def __init__(self, output_str, subject=None, output_dir=None, settings_file=None, run=None, eyetracker_on=False):
        print(settings_file)
        super().__init__(output_str, subject=subject,
                         output_dir=output_dir, settings_file=settings_file, run=run, eyetracker_on=eyetracker_on)

        logging.warn(self.settings['run'])

    def create_trials(self):

        txt = """
        In this task, you will see two piles of swiss Franc coins in
        succession. Both piles are combined with a pie chart in.
        The part of the pie chart that is lightly colored indicates
        the probability of a lottery you will gain the amount of
        Swiss Francs represented by the pile.
        There is always one pile that has a probability of 100% for payout.
        The other probability changes every 16 trials.

        Your task is to either select the first lottery or
        the second lottery, by using your index or middle finger.
        Immediately after your choice, we ask how certain you were
        about your choice from a scale from 1 (very uncertain)
        to 4 (very certain).

        This is run {run}/{self.n_runs}.

        Press any of your buttons to continue.

        """

        task_settings_folder = op.abspath(op.join('settings', 'task'))
        fn = op.abspath(op.join(task_settings_folder,
                                f'sub-{self.subject}_ses-task.tsv'))

        settings = pd.read_table(fn)

        if self.settings['run'] is not None:
            settings = settings.set_index(['run'])
            settings = settings.loc[int(self.settings['run'])]

        print(settings)

        jitter1 = self.settings['task'].get('jitter1')
        jitter2 = self.settings['task'].get('jitter2')

        jitter1 = np.repeat(jitter1, np.ceil(len(settings)/len(jitter1)))
        jitter2 = np.repeat(jitter2, np.ceil(len(settings)/len(jitter2)))

        np.random.shuffle(jitter1)
        np.random.shuffle(jitter2)

        self.n_runs = settings.run.unique().shape[0]

        settings['jitter1'] = jitter1[:len(settings)]
        settings['jitter2'] = jitter2[:len(settings)]

        self.trials = []

        for run, d in settings.groupby(['run'], sort=False):
            self.trials.append(GambleInstructionTrial(self, trial_nr=run,
                                                      n_runs=self.n_runs,
                                                      run=run))
            for (p1, p2), d2 in d.groupby(['p1', 'p2'], sort=False):
                n_trials_in_miniblock = len(d2)
                self.trials.append(IntroBlockTrial(session=self, trial_nr=0,
                                                   n_trials=n_trials_in_miniblock,
                                                   prob1=p1,
                                                   prob2=p2))


                for ix, row in d2.iterrows():
                    self.trials.append(GambleTrial(self, row.trial,
                                                   prob1=row.p1, prob2=row.p2,
                                                   num1=int(row.n1),
                                                   num2=int(row.n2),
                                                   jitter1=row.jitter1,
                                                   jitter2=row.jitter2))

        

class TaskSessionMRI(TaskSession):

    def create_trials(self):

        super().create_trials()

        n_dummies = self.settings['mri'].get('n_dummy_scans')

        self.trials.insert(1, DummyWaiterTrial(session=self, n_triggers=n_dummies, trial_nr=0))

        self.trials.append(OutroTrial(self, -1, phase_durations=[np.inf]))

if __name__ == '__main__':

    session_cls = TaskSessionMRI
    task = 'task'
    run_experiment(session_cls, task=task)
