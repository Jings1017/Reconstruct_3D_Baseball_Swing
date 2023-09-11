import cv2
import os

# 設置視頻文件名和視頻編解碼器
output_video_name = 'cam3-1_line_v2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4編解碼器，也可以使用其他編解碼器

# 設置視頻寬度和高度（根據您的圖像幀大小調整）
frame_width = 1920
frame_height = 1080

# 初始化VideoWriter對象
out = cv2.VideoWriter(output_video_name, fourcc, 15.0, (frame_width, frame_height))

# 輸入圖像幀的文件夾路徑
frame_folder = './segment/out/nthu_swing_0920/cam3-1_frames'

# 檢查文件夾是否存在
if not os.path.exists(frame_folder):
    print(f"FOLDER '{frame_folder}' is not exist")
else:
    # 列出文件夾中的所有圖像文件
    frame_files = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.png')]
    frame_files.sort()

    # 遍歷圖像文件，將每個圖像幀添加到視頻中
    for frame_file in frame_files:
        if 'line' in frame_file:
            print(frame_file)
            frame = cv2.imread(frame_file)
            out.write(frame)

    # 釋放VideoWriter對象
    out.release()
    print(f"Video '{output_video_name}' is Created !")
