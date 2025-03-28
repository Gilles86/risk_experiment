from stimuli import FixationLines
from exptools2.core import Session
from psychopy import visual, logging
import pandas as pd
import os.path as op
from trial import InstructionTrial, OutroTrial, TaskInstructionTrial, IntroBlockTrial, GambleTrial
import numpy as np


class NumeralRiskSession(Session):
    """Session class for the numeral risk task."""

    def __init__(self, output_str, subject=None, output_dir=None, settings_file=None, run=None):
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file)

        self.subject = subject
        self.settings['run'] = run

        self.fixation_dot = visual.Circle(
            self.win,
            radius=self.settings['various'].get('fixation_size', .1),
            fillColor='white',
            lineColor=None,
            colorSpace='rgb')
        

        logging.warn(self.settings['run'])

    def _create_logfile(self):
        """Creates a logfile."""
        log_path = op.join(self.output_dir, self.output_str + '_log.txt')
        return logging.LogFile(f=log_path, filemode='w', level=logging.WARNING)

    def run(self):
        """Runs the experiment."""
        self.start_experiment()

        for trial in self.trials:
            trial.run()


        # Load and filter the log
        choice_log = self.global_log[self.global_log['event_type'] == 'choice']

        # Randomly sample a trial
        trial = choice_log.sample(1).iloc[0]

        import random

        def get_text_for_sampled_trial(trial):
            if pd.isnull(trial['choice']):
                return (
                    "On the selected trial, you did not respond within the 2.5-second time limit. "
                    "Therefore, you will not receive any additional monetary bonus for this trial.\n\n"
                    "Note: This does not affect your fixed compensation of CHF 30 per hour."
                )

            # Determine which option was risky and what was chosen
            risky_first = trial['prob1'] != 1.0
            chose_risky = (trial['choice'] == 1) == risky_first
            n_risky = trial['n1'] if risky_first else trial['n2']
            n_safe = trial['n2'] if risky_first else trial['n1']

            if chose_risky:
                # Simulate the digital roll between 0 and 99
                roll = random.randint(0, 99)
                won = roll <= 54
                bonus = n_risky if won else 0.00

                return (
                    f"On the selected trial, you chose the risky option.\n\n"
                    f"You had a 55% chance of winning CHF {n_risky:.2f}, and a 45% chance of winning CHF 0.00.\n\n"
                    "A random number between 00 and 99 has now been drawn to determine the outcome...\n\n"
                    f"You get the bonus if the number drawn is 00-54.\n\n"
                    f"**The number drawn was: {roll:02d}**\n\n"
                    f"You {'won' if won else 'did not win'} this gamble.\n"
                    f"You will receive CHF {bonus:.2f} as a bonus.\n\n"
                    "Note: The potential bonus is in addition to your fixed compensation of CHF 30 per hour."
                )
            else:
                return (
                    f"On the selected trial, you chose the safe option.\n"
                    f"You will now receive CHF {n_safe:.2f} as a bonus.\n\n"
                    "Note: This is in addition to your fixed compensation of CHF 30 per hour."
                )

        txt = get_text_for_sampled_trial(trial)
        print(txt)
        logging.warning(txt)

        outcome_trial = InstructionTrial(self, trial_nr=-10000, txt=txt, phase_durations=[np.inf],
                                         allow_keypresses=False)
        outcome_trial.run()

        self.close()

    def create_trials(self, n_runs=None, n_trials_per_run=None):

        if n_runs is None:
            n_runs = self.settings['task'].get('n_runs', 4)
            n_trials_per_run = self.settings['task'].get('n_trials_per_run', 32)

        def create_run_settings(n_trials, order):
            min_safe_payoff = self.settings['payoffs'].get('min_safe_payoff', 5)
            max_safe_payoff = self.settings['payoffs'].get('max_safe_payoff', 28)
            min_risky_payoff_frac = self.settings['payoffs'].get('min_risky_frac', 1)
            max_risky_payoff_frac = self.settings['payoffs'].get('max_risky_frac', 4)

            # Safe payoff sampled uniformly in log space
            log_min = np.log(min_safe_payoff)
            log_max = np.log(max_safe_payoff)
            n_safe = np.exp(np.random.rand(n_trials) * (log_max - log_min) + log_min)

            frac = np.random.rand(n_trials) * (max_risky_payoff_frac - min_risky_payoff_frac) + min_risky_payoff_frac
            n_risky = n_safe * frac

            jitter1 = self.settings['task'].get('jitter1')
            jitter2 = self.settings['task'].get('jitter2')

            jitter1 = np.repeat(jitter1, np.ceil(n_trials / len(jitter1)))[:n_trials]
            jitter2 = np.repeat(jitter2, np.ceil(n_trials / len(jitter2)))[:n_trials]

            np.random.shuffle(jitter1)
            np.random.shuffle(jitter2)

            p1 = np.array([.55, 1.0]) if order == 0 else np.array([1.0, .55])
            p1 = np.tile(p1, int(np.ceil(n_trials / 2)))[:n_trials]
            p2 = np.where(p1 == 1.0, .55, 1.0)

            risky_first = p1 != 1.0

            settings = pd.DataFrame({
                'p1': p1,
                'p2': p2,
                'n_safe': n_safe,
                'n_risky': n_risky,
                'jitter1': jitter1,
                'jitter2': jitter2
            })

            settings['n1'] = settings['n_risky'].where(risky_first, settings['n_safe'])
            settings['n2'] = settings['n_risky'].where(~risky_first, settings['n_safe'])
            settings['trial_nr'] = np.arange(n_trials) + 1 + (n_runs - 1) * n_trials_per_run

            return settings

        runs = list(range(1, n_runs + 1))
        risky_order = np.tile([0, 1], n_runs // 2 + 1)[:n_runs]
        np.random.shuffle(risky_order)
        settings = pd.concat(
            [create_run_settings(n_trials_per_run, order) for _, order in zip(runs, risky_order)],
            keys=runs,
            names=['run']
        )
        
        logging.warning(settings)

        self.trials = []

        for run, d in settings.groupby('run', sort=False):
            self.trials.append(TaskInstructionTrial(self, trial_nr=-run,
                                                    n_runs=n_runs,
                                                    run=run))
            for (p1, p2), d2 in d.groupby(['p1', 'p2'], sort=False):
                n_trials_in_miniblock = len(d2)
                self.trials.append(IntroBlockTrial(session=self, trial_nr=0,
                                                   n_trials=n_trials_in_miniblock,
                                                   prob1=p1, prob2=p2))
                for _, row in d2.iterrows():
                    self.trials.append(GambleTrial(self, trial_nr=row.trial_nr,
                                                   run=run,
                                                   prob1=row.p1, prob2=row.p2,
                                                   num1=np.round(row.n1, 2),
                                                   num2=np.round(row.n2, 2),
                                                   jitter1=row.jitter1,
                                                   jitter2=row.jitter2))

        txt = "You have completed the task. Now press a button to finish the experiment" \
                "and see whether  you got a bonus!"

        outro_trial = InstructionTrial(self, trial_nr=-1000, txt=txt, phase_durations=[np.inf])
        self.trials.append(outro_trial)
