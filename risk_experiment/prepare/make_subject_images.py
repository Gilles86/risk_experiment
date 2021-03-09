import argparse
from bids import BIDSLayout
from nilearn import image
import numpy as np
import os
import os.path as op
from nilearn import plotting
import matplotlib.pyplot as plt


def main(subject, bids_folder):

    output_folder = op.join(bids_folder, 'derivatives',
                            'subject_image', f'sub-{subject}')

    if not op.exists(output_folder):
        os.makedirs(output_folder)

    layout = BIDSLayout(bids_folder, validate=False)

    t1w_3t = layout.get(subject=subject, suffix='T1w', session='3t1')[0].path

    t1w_7t = layout.get(subject=subject, suffix='T1w', session='7t1')
    t1w_7t = image.concat_imgs([im.path for im in t1w_7t])
    t1w_7t = image.mean_img(t1w_7t)

    t2starw_7t = layout.get(subject=subject, suffix='T2starw', session='7t1',
            acquisition='average')[0].path
    print(t2starw_7t)


    def plot_img(im, label):

        if 'T2starw' in label:
            bounds = ((-65, 65), (-80, 80), (-20, 10))
        else:
            bounds = ((-65, 65), (-65, 65), (-30, 60))

        for bnd, display_mode in zip(bounds[2:], ['x', 'y', 'z'][2:]):
            print(bnd, display_mode)
            for coord in np.arange(bnd[0], bnd[1] + 5, 5):
                fn = op.join(
                    output_folder, f'sub-{subject}_label-{label}_axis-{display_mode}_coord-{coord}.png')
                plotting.plot_img(im, cut_coords=[coord],
                                  display_mode=display_mode,
                                  cmap='gray')
                fig = plt.gcf()
                fig.set_dpi(500)
                fig.savefig(fn)
                


    # plot_img(t1w_3t, label='3T_t1w')
    plot_img(t1w_7t, label='7T_t1w')
    # plot_img(t2starw_7t, label='7T_T2starw')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/data2/ds-risk')

    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder)
