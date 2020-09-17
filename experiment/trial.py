import numpy as np
from exptools2.core import Trial
from psychopy.visual import TextStim
from stimuli import FixationLines


class InstructionTrial(Trial):
    """ Simple trial with instruction text. """

    def __init__(self, session, trial_nr, phase_durations=[np.inf],
                 txt=None, keys=None, **kwargs):

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        txt_height = self.session.settings['various'].get('text_height')
        txt_width = self.session.settings['various'].get('text_width')

        if txt is None:
            txt = '''Pess any button to continue.'''

        self.text = TextStim(self.session.win, txt,
                             height=txt_height, wrapWidth=txt_width, **kwargs)

        self.keys = keys

    def draw(self):
        self.text.draw()

    def get_events(self):
        events = super().get_events()

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
        self.session.fixation_lines.draw()
        super().draw()

    def get_events(self):
        events = Trial.get_events(self)

        if events:
            for key, t in events:
                if key == 'space':
                    self.stop_phase()
