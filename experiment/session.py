from stimuli import FixationLines
from exptools2.core import Session, PylinkEyetrackerSession
from psychopy import visual, logging
import pandas as pd
import os.path as op


class PileSession(PylinkEyetrackerSession):
    """ Simple session with x trials. """

    def __init__(self, output_str, subject=None, output_dir=None, settings_file=None, run=None, eyetracker_on=True):
        """ Initializes TestSession object. """
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file, eyetracker_on=eyetracker_on)
        self.subject = subject
        self.settings['run'] = run
        print(self.settings)

        self.fixation_lines = FixationLines(self.win,
                                            self.settings['pile'].get(
                                                'aperture_radius')*2,
                                            color=(1, -1, -1))

        self.image1 = visual.ImageStim(self.win,
                                       self.settings['pile'].get('image1'),
                                       texRes=32,
                                       size=self.settings['pile'].get('dot_radius'))

    def _create_logfile(self):
        """ Creates a logfile. """
        log_path = op.join(self.output_dir, self.output_str + '_log.txt')
        return logging.LogFile(f=log_path, filemode='w', level=logging.WARNING)

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

    def close(self):
        super().close()

        array_log = []

        for trial in self.trials:
            if isinstance(trial, self.Trial):
                array_log.append(trial.get_stimulus_array_log())

        if len(array_log) > 0:
            array_log = pd.concat(array_log)

            array_log.to_csv(op.join(
                self.output_dir, self.output_str + '_stimarray_locations.tsv'), sep='\t')


class PileEyeTrackerSession(PileSession, PylinkEyetrackerSession):
    pass

