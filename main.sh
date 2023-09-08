VIDEONAME="1"
VIDEOFOLDER="input/baseball_swing_20221122/1"

python src/main.py \
--target_view1 "${VIDEOFOLDER}/${VIDEONAME}_view1_target.npy" \
--target_view2 "${VIDEOFOLDER}/${VIDEONAME}_view2_target.npy" \
--ref_points_view1 "${VIDEOFOLDER}/${VIDEONAME}_view1_coordinate.npy" \
--ref_points_view2 "${VIDEOFOLDER}/${VIDEONAME}_view2_coordinate.npy" \
--calib_file "${VIDEOFOLDER}/calib.npz" \
--start_frame 0 \
--end_frame 1000 \
--frame_skip 1 \
--show \
--input_video_1 "${VIDEOFOLDER}/${VIDEONAME}_view1.MP4" \
--input_video_2 "${VIDEOFOLDER}/${VIDEONAME}_view2.MP4" \
--output_video "output/${VIDEONAME}/${VIDEONAME}.mp4"