import numpy as np
from exptools2.core import Trial
from psychopy.visual import TextStim

class InstructionTrial(Trial):
    """ Simple trial with instruction text. """
    def __init__(self, session, trial_nr, phase_durations=[np.inf],
            txt=None, **kwargs):

        txt_height = kwargs.pop('height', None)
        txt_width = kwargs.pop('wrapWidth', None)

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        if txt is None:
            txt = '''Pess any button to continue.'''

        self.text = TextStim(self.session.win, txt,
                height=txt_height, wrapWidth=txt_width, **kwargs)

        self.n_triggers = 0

    def draw(self):
        self.text.draw()

    def get_events(self):
        events = super().get_events()

        if events:
            self.stop_phase()

class DummyWaiterTrial(InstructionTrial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations=None, n_triggers=1,
            txt="Waiting for scanner triggers.", **kwargs):
        phase_durations = [np.inf] * n_triggers

        super().__init__(session, trial_nr, phase_durations, txt, **kwargs)


    def get_events(self):
        events = super().get_events()

        if events:
            for key, t in events:
                if key == self.session.mri_trigger:
                    self.stop_phase()
