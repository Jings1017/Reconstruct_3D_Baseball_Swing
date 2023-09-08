import numpy as np
import argparse
import os
from reconstruct import Reconstruct3D

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--end_frame", default=None, help="End frame", type=int)
    parser.add_argument("-s", "--start_frame", default=None, help="Start frame", type=int)
    parser.add_argument("--frame_skip", default=0, help="Start frame", type=int)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--calib_file", type=str)
    parser.add_argument("--ref_points_view1", type=str)
    parser.add_argument("--ref_points_view2", type=str)
    parser.add_argument("--target_view1", type=str)
    parser.add_argument("--target_view2", type=str)
    parser.add_argument("--output_video", type=str)
    parser.add_argument("--input_video_1", type=str)
    parser.add_argument("--input_video_2", type=str)

    args = parser.parse_args()
    end_frame = args.end_frame
    start_frame = args.start_frame
    frame_skip = args.frame_skip
    ref_points_view1 = args.ref_points_view1
    ref_points_view2 = args.ref_points_view2
    calibration = args.calib_file
    output_video = args.output_video
    input_video_1 = args.input_video_1
    input_video_2 = args.input_video_2

    out_dir = os.path.dirname(output_video)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    target_view1 = np.load(args.target_view1, allow_pickle=True)
    target_view2 = np.load(args.target_view2, allow_pickle=True)

    print(len(target_view1), len(target_view2))
    if len(target_view1) < len(target_view2):
        target_view2 = target_view2[: len(target_view1)]
    else:
        target_view1 = target_view1[: len(target_view2)]

    recon = Reconstruct3D()
    balls_frame_num, court = recon.reconstruct(
        start_frame,
        end_frame,
        frame_skip,
        target_view1,
        target_view2,
        ref_points_view1,
        ref_points_view2,
        calibration,
        input_video_1,
        input_video_2,
        output_fig=output_video,
        show=args.show,
    )
