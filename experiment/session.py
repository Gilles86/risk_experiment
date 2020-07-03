from stimuli import FixationLines
from exptools2.core import Session, PylinkEyetrackerSession
from psychopy import visual

class PileSession(PylinkEyetrackerSession):
    """ Simple session with x trials. """
    def __init__(self, output_str, subject=None, output_dir=None, settings_file=None, eyetracker_on=True):
        """ Initializes TestSession object. """
        super().__init__(output_str, output_dir=None, settings_file=settings_file, eyetracker_on=eyetracker_on)
        self.subject = subject
        self.use_eyetracker = eyetracker_on

        print(self.settings)

        self.fixation_lines = FixationLines(self.win,
                self.settings['pile'].get('aperture_radius')*2,
                color=(1, -1, -1))
        
        self.image1 = visual.ImageStim(self.win, 
                self.settings['pile'].get('image1'),
                texRes=32,
                size=self.settings['pile'].get('dot_radius')*2)

    def run(self):
        """ Runs experiment. """
        if self.eyetracker_on:
            self.calibrate_eyetracker()

        self.start_experiment()

        if self.eyetracker_on:
            self.start_recording_eyetracker()
        for trial in self.trials:
            trial.run()

        self.close()
