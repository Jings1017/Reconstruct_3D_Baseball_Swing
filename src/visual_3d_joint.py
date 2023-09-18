import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

class Visual3DJoint:
    def __init__(self) -> None:
        pass

    def set_saved_video(self, output_video, size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        video = cv2.VideoWriter(output_video, fourcc, fps, size)
        return video

    def drawCourt(self, court_category, ax):
        if court_category == "baseball_bat":
            points = np.array(
                [
                    [0, 0, 0],
                    [0, 0, 27],
                    [0, 35, 27],
                    [0, 35, 0],
                    [46, 0, 0],
                    [46, 0, 27],
                    [46, 35, 27],
                    [46, 35, 0],
                ]
            )
            court_edges = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4]
            curves = points[court_edges]
            ax.plot(curves[:, 0], curves[:, 1], curves[:, 2], c="k", linewidth=1)
            court_edges = [1, 5]
            curves = points[court_edges]
            ax.plot(curves[:, 0], curves[:, 1], curves[:, 2], c="k", linewidth=1)
            court_edges = [2, 6]
            curves = points[court_edges]
            ax.plot(curves[:, 0], curves[:, 1], curves[:, 2], c="k", linewidth=1)
            court_edges = [3, 7]
            curves = points[court_edges]
            ax.plot(curves[:, 0], curves[:, 1], curves[:, 2], c="k", linewidth=1)
        return ax
    
    def draw_line(self, ax, joint_target_x, joint_target_y, joint_target_z, frame_count, id1, id2, color):
        ax.plot(
                [joint_target_x[frame_count][id1], joint_target_x[frame_count][id2]],
                [joint_target_y[frame_count][id1], joint_target_y[frame_count][id2]],
                [joint_target_z[frame_count][id1], joint_target_z[frame_count][id2]],
                color=color,
                linewidth=3
        )


    def show_3D(
        self,
        input_video1,
        input_video2,
        court,
        target,
        frame_skip,
        end_frame,
        alpha=0.2,
        add_court=False,
        save_name=None,
        is_set_lim=True,
        court_category="baseball_bat",
    ):
        if save_name is not None:
            video = self.set_saved_video(save_name, (1224, 648))

        court_x, court_y, court_z = court[0], court[1], court[2]
        joint_target_x, joint_target_y, joint_target_z = target[0], target[1], target[2]
        frame_count = -1
        count = -1
        cap = cv2.VideoCapture(input_video1)
        cap2 = cv2.VideoCapture(input_video2)

        print(joint_target_x.shape)

        while cap.isOpened():
            fig = plt.figure(dpi=300)
            fig.set_size_inches(7.2, 7.2)
            gs = gridspec.GridSpec(6, 6)
            ax = plt.subplot(gs[:, :], projection="3d")

            if is_set_lim:
                ax.set_xlim(-50, 350)
                ax.set_ylim(-200, 100)
                ax.set_zlim(0, 300)

            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("y", fontsize=12)
            ax.set_zlabel("z", fontsize=12)
            frame_count += 1
            if frame_count == end_frame:
                break
            ret, frame = cap.read()
            ret2, frame2 = cap2.read()
            if frame_count % frame_skip != 0:
                continue
            else:
                count += 1

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            ax.scatter(court_x, court_y, court_z, marker="o", alpha=alpha)
 
            if add_court:
                self.drawCourt(court_category, ax)

            ax.scatter(joint_target_x[frame_count][0], joint_target_y[frame_count][0], joint_target_z[frame_count][0], color='black')
            ax.scatter(joint_target_x[frame_count][11:17], joint_target_y[frame_count][11:17], joint_target_z[frame_count][11:17], color='black')
            ax.scatter(joint_target_x[frame_count][23:29], joint_target_y[frame_count][23:29], joint_target_z[frame_count][23:29], color='black')

            # drawing 
            # body
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 11, 12, 'red')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 11, 23, 'red')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 12, 24, 'red')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 23, 24, 'red')

            # arm
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 12, 14, 'blue')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 14, 16, 'blue')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 11, 13, 'green')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 13, 15, 'green')

            # leg
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 24, 26, 'orange')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 26, 28, 'orange')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 23, 25, 'violet')
            self.draw_line(ax, joint_target_x, joint_target_y, joint_target_z, frame_count, 25, 27, 'violet')



            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frames = cv2.vconcat([frame, frame2])
            merge_image = cv2.hconcat([frames, img])
            resize_size = (int(merge_image.shape[1] * 0.3), int(merge_image.shape[0] * 0.3))
            merge_image = cv2.resize(merge_image, resize_size)
            
            cv2.imshow('check frame', merge_image)
            if save_name is not None:
                video.write(merge_image)
            if cv2.waitKey(1) == ord("q"):
                break
            plt.close(fig)

        if save_name is not None:
            video.release()
        cap.release()
        cap2.release()
        cv2.destroyAllWindows()
        return
