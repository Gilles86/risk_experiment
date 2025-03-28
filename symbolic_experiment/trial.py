from exptools2.core import Session
from exptools2.core import Trial
from psychopy.visual import Pie, TextStim
from stimuli import _create_stimulus_array, ProbabilityPieChart
import numpy as np
import os.path as op
import pandas as pd

class IntroBlockTrial(Trial):

    def __init__(self, session, trial_nr, phase_durations=[5.],
                 prob1=0.55, prob2=1.0, n_trials=16, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)

        txt = f"""
        In this block of {n_trials} trials, the first option will have a
        winning chance of {int(prob1*100):d}%.\n\n
        The second option will have a winning chance of {int(prob2*100):d}%.
        """

        text_width = self.session.settings['various'].get('text_width')
        text_height = self.session.settings['various'].get('text_height')
        piechart_width = self.session.settings['various'].get('piechart_width')

        piechart_pos1 = .5 * -text_width - .25 * piechart_width, 1.5 * piechart_width
        piechart_pos2 = .5 * -text_width - .25 * piechart_width, -1.5 * piechart_width

        self.piechart1 = ProbabilityPieChart(
            self.session.win, prob1, pos=piechart_pos1, size=piechart_width)
        self.piechart2 = ProbabilityPieChart(
            self.session.win, prob2, pos=piechart_pos2, size=piechart_width)

        self.text = TextStim(self.session.win, txt,
                             wrapWidth=text_width,
                             height=text_height,
                             colorSpace='rgb',)

    def draw(self):
        self.text.draw()
        self.piechart1.draw()
        self.piechart2.draw()


class GambleTrial(Trial):
    def __init__(self, session, run, trial_nr, phase_durations=None,
                 prob1=0.55, prob2=1.0, num1=10, num2=5,
                 jitter1=2.5, jitter2=4.0, **kwargs):

        self.max_response_time = session.settings['various'].get('max_response_time', 2.5)

        if phase_durations is None:
            phase_durations = [.25, .3, .3, .5, .6, jitter1, .3, .3, .6, self.max_response_time, jitter2 - self.max_response_time]
        elif len(phase_durations) == 12:
            phase_durations = phase_durations
        else:
            raise Exception(
                "Don't directly set phase_durations for GambleTrial!")

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.parameters['run'] = run
        self.parameters['prob1'] = prob1
        self.parameters['prob2'] = prob2

        self.parameters['n1'] = num1
        self.parameters['n2'] = num2

        self.buttons = self.session.settings['various'].get('buttons')
        piechart_width = self.session.settings['various'].get('piechart_width')

        self.piechart1 = ProbabilityPieChart(
            self.session.win, prob=prob1, size=piechart_width)
        self.piechart2 = ProbabilityPieChart(
            self.session.win, prob=prob2, size=piechart_width)

        # self.text1 = TextStim('Pie1: {int(

        self.payoff1_text = TextStim(self.session.win, text=f'CHF {num1:0.2f}',
                                     pos=(0, 0),
                                     height=self.session.settings['payoffs'].get('text_height', 5.25),
                                     colorSpace='rgb',)

        self.payoff2_text = TextStim(self.session.win, text=f'CHF {num2:0.2f}',
                                     pos=(0, 0),
                                     height=self.session.settings['payoffs'].get('text_height', 5.25),
                                     colorSpace='rgb',)


        self.choice_stim = TextStim(self.session.win, colorSpace='rgb',)

        self.choice = None

        self.last_key_responses = dict(zip(self.buttons + [self.session.mri_trigger], [0.0] * 5))

    def draw(self):



        if self.phase == 0:
            self.session.fixation_dot.setColor((1, -1, -1))
            self.session.fixation_dot.draw()
        elif self.phase == 1:
            self.session.fixation_dot.setColor((1, 1, 1))
            self.session.fixation_dot.draw()
        elif self.phase == 2:
            self.piechart1.draw()
        elif self.phase == 3:
            self.session.fixation_dot.draw()
        elif self.phase == 4:
            self.payoff1_text.draw()
        elif self.phase == 5:
            self.session.fixation_dot.draw()
        elif self.phase == 6:
            self.piechart2.draw()
        elif self.phase == 8:
            self.payoff2_text.draw()

        elif self.phase == 9:
            if self.choice is not None:
                if (self.session.clock.getTime() - self.choice_time) < self.session.settings['various'].get('feedback_duration', 1.0):
                    self.choice_stim.draw()
                else:
                    self.stop_phase()
            else:
                self.session.fixation_dot.draw()

        elif self.phase == 10:
            if self.choice is None:
                self.choice_stim.color = 'red'
                self.choice_stim.text = 'Too slow!'
                self.choice_stim.draw()
            else:
                if (self.session.clock.getTime() - self.choice_time) < self.session.settings['various'].get('feedback_duration', 1.0):
                    self.choice_stim.draw()
                else:
                    self.stop_phase()
        


    def get_events(self):
        events = super().get_events()

        for key, t in events:
            if key not in self.last_key_responses:
                self.last_key_responses[key] = t - 0.6

            if t - self.last_key_responses[key] > 0.5:
                if (self.phase > 7) & (self.phase < 10):
                    if self.choice is None:
                        if key in [self.buttons[0], self.buttons[1]]:
                            self.choice_time = self.session.clock.getTime()
                            self.response_time = self.choice_time - self.stimulus_presentation_time
                            self.parameters['response_time'] = self.response_time
                            
                            if key == self.buttons[0]:
                                self.choice = 1
                            elif key == self.buttons[1]:
                                self.choice = 2

                            self.choice_stim.text = f'{self.choice}'

                            self.log(choice=self.choice)

            self.last_key_responses[key] = t

        return events

    def log(self, choice=None):

        onset = self.session.clock.getTime()
        idx = self.session.global_log.shape[0]
        self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
        self.session.global_log.loc[idx, 'onset'] = onset
        self.session.global_log.loc[idx, 'phase'] = self.phase
        self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames
        self.session.global_log.loc[idx, 'event_type'] = 'choice'
        self.session.global_log.loc[idx, 'choice'] = choice

        for key in self.parameters:
            self.session.global_log.loc[idx, key] = self.parameters[key]

    def log_phase_info(self, phase=None):
        if phase == 7:
            self.stimulus_presentation_time = self.session.clock.getTime()

        elif self.phase == 10:
            if self.choice is None:
                self.log(choice=None)
        
        super().log_phase_info(phase)


class InstructionTrial(Trial):
    """ Simple trial with instruction text. """

    def __init__(self, session, trial_nr, phase_durations=[np.inf],
                 allow_keypresses=True,
                 txt=None, keys=None, **kwargs):

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.allow_keypresses = allow_keypresses

        txt_height = self.session.settings['various'].get('text_height', 0.05)
        txt_width = self.session.settings['various'].get('text_width', 1.4)  # wider if left-aligned

        if txt is None:
            txt = "Press any button to continue."

        self.text = TextStim(
            self.session.win,
            text=txt,
            height=txt_height,
            wrapWidth=txt_width,
            alignText='left',         # ← left-aligns the text
            anchorHoriz='left',       # ← makes alignment work as expected
            pos=(-txt_width / 2, 0),  # ← so it starts at left side of screen
            colorSpace='rgb',
            **kwargs
        )

        self.keys = keys

    def draw(self):
        self.text.draw()

    def get_events(self):
        events = super().get_events()

        if self.allow_keypresses:
            if self.keys is None:
                if events:
                    self.stop_phase()
            else:
                for key, t in events:
                    if key in self.keys:
                        self.stop_phase()

class DummyWaiterTrial(InstructionTrial):
    """ Simple trial with text (trial x) and fixation. """

    def __init__(self, session, trial_nr, phase_durations=None, n_triggers=1,
                 txt="Waiting for scanner triggers.", **kwargs):
        phase_durations = [np.inf] * n_triggers

        super().__init__(session, trial_nr, phase_durations, txt, **kwargs)

        self.last_trigger = 0.0

    def get_events(self):
        events = Trial.get_events(self)

        if events:
            for key, t in events:
                if key == self.session.mri_trigger:
                    if t - self.last_trigger > .5:
                        self.stop_phase()
                        self.last_trigger = t

class OutroTrial(InstructionTrial):
    """ Simple trial with only fixation cross.  """

    def __init__(self, session, trial_nr, phase_durations, **kwargs):

        txt = '''Please lie still for a few moments.'''
        super().__init__(session, trial_nr, phase_durations, txt=txt, **kwargs)

    def draw(self):
        self.session.fixation_dot.draw()
        super().draw()

    def get_events(self):
        events = Trial.get_events(self)

        if events:
            for key, t in events:
                if key == 'space':
                    self.stop_phase()

class TaskInstructionTrial(InstructionTrial):
    
    def __init__(self, session, trial_nr, run, txt=None, n_runs=3, phase_durations=[np.inf],
                 **kwargs):

        txt = f"""
        This is run {run} of {n_runs}.

        In this task, you will be presented with two options, one after the other.
        One option is a safe option — a guaranteed amount of money.
        The other option is a risky gamble with a 55% chance of winning.

        You will have to choose between these two options using the keyboard:
        - Press **j** to select the first option presented.
        - Press **k** to select the second option presented.

        **Note:** If you respond too late or do not respond at all, you will not receive any money for that trial.

        Feel free to take a short break now if you'd like.

        Press either button (j or k) to continue.
        """


        super().__init__(session=session, trial_nr=trial_nr, phase_durations=phase_durations, txt=txt, **kwargs)