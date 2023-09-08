def draw_target_point(img, targets):
    circle_color = (255, 0, 0)
    size = 5
    if len(targets) == 2:
        img = cv2.circle(img, targets[0], size, circle_color, -1)
        img = cv2.circle(img, targets[1], size, circle_color, -1)
        img = cv2.line(img, targets[0], targets[1], circle_color, size)

    return img


if __name__ == "__main__":
    import cv2
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, help="input video")
    parser.add_argument("--input_coordinate", type=str, help="input video")
    parser.add_argument("--start", type=int, help="start frame", default=0)
    parser.add_argument("--end", type=int, help="end frame", default=10000000)

    args = parser.parse_args()
    input_video = args.input_video
    target_view1 = np.load(args.input_coordinate, allow_pickle=True)
    start = args.start
    end = args.end
    # play video
    cap = cv2.VideoCapture(input_video)
    count = -1
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        if count <= start:
            continue
        if count > end:
            break
        if ret:
            frame = draw_target_point(frame, target_view1[count])
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == ord("q") or key == 27:  # q or esc
                break

    cv2.destroyAllWindows()
