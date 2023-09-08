if __name__ == "__main__":
    import cv2
    import argparse
    from os import makedirs
    from os.path import exists
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input video")
    parser.add_argument("--output", type=str, help="output frame folder")
    args = parser.parse_args()

    print("Video input: ", args.input)
    video_path = args.input
    output_folder = args.output
    file_name = Path(video_path).stem
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    if not exists(output_folder + "/" + file_name):
        makedirs(output_folder + "/" + file_name)
    while success:
        cv2.imwrite(
            output_folder + "/" + file_name + "/frame%d.png" % count, image
        )  # save frame as png file
        success, image = vidcap.read()
        print("Read a new frame: ", success)
        count += 1
    print("total frame: {}".format(count))
    print("Saving to: ", args.output)
