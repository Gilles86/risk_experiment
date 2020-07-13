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

        n_dummies = self.settings['mri'].get('n_dummy_scans')
        phase_durations = [np.inf] * (n_dummies + 1)

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

        This is run {run}/3.

        Press any of your buttons to continue.

        """

        task_settings_folder = op.abspath(op.join('settings', 'task'))
        fn = op.abspath(op.join(task_settings_folder,
                                f'sub-{self.subject}_ses-task.tsv'))
        settings = pd.read_table(fn)

        print(settings)
        settings = settings.set_index(['run'])
        print(settings)
        settings = settings.loc[int(self.settings['run'])]
        print(settings)

        n_dummies = self.settings['mri'].get('n_dummy_scans')

        self.trials = [GambleInstructionTrial(self, trial_nr=0, run=self.settings['run'])]

        self.trials.append(DummyWaiterTrial(
            session=self, n_triggers=n_dummies, trial_nr=0))

        jitter1 = np.repeat([5,6,7,8], 128/4)
        jitter2 = np.repeat(np.linspace(4, 6, 4), 128/4)
        np.random.shuffle(jitter1)
        np.random.shuffle(jitter2)

        for j1, j2, (ix, row) in zip(jitter1, jitter2, settings.iterrows()):
            self.trials.append(GambleTrial(self, row.trial,
                                           prob1=row.p1, prob2=row.p2,
                                           num1=int(row.n1),
                                           num2=int(row.n2),
                                           jitter1=j1,
                                           jitter2=j2))

        


if __name__ == '__main__':

    session_cls = TaskSession
    task = 'task'
    run_experiment(session_cls, task=task)
