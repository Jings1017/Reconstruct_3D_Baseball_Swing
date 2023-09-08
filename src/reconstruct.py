import numpy as np
import cv2
from visual3d import Visual3D

class Reconstruct3D:
    def __init__(self) -> None:
        pass

    def reconstruct(
        self,
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
        output_fig=None,
        show=False,
    ):
        """
        output
            balls_frame_num: e.g., [[x, y ,z, frame number], [x, y ,z, frame number],...]
            court: e.g., [[x1, y1, z1], [x2, y2, z2],...]
        """
        court = None
        # if has start an end frame
        if start_frame is not None:
            print("Start from frame %d" % start_frame)
            target_view1 = target_view1[start_frame:]
            target_view2 = target_view2[start_frame:]

        if end_frame is not None:
            print("Run %d frames." % end_frame)
            target_view1 = target_view1[:end_frame]
            target_view2 = target_view2[:end_frame]

        all_target_view1 = target_view1
        all_target_view2 = target_view2
        frames = zip(target_view1, target_view2)

        target_frame_num = []

        """
        yolo[i]
            list([]) or list([[68, ['44', '973', '36', '30']]])
        detectron
            list([]) or list([(1330, 292), (1264, 430)])
        """
        proj_map_1, proj_map_2 = self.project_points(
            src_points_1=self.get_ref_points(ref_points_view1),
            src_points_2=self.get_ref_points(ref_points_view2),
            dst_points=self.get_ref_points(),
            dist=np.load(calibration)["dist_coefs"],
            mtx=np.load(calibration)["camera_matrix"],
        )

        for frame_num, frame in enumerate(frames):
            if frame_num % frame_skip != 0:
                continue
            view1, view2 = frame
            # print(view1, view2)
            # get targets position
            if len(view1) > 0 and len(view2) > 0:
                target_view1 = np.array(view1, dtype="int")
                target_view2 = np.array(view2, dtype="int")

                court, t1, t2 = self.draw2court(
                    target_view1=target_view1,
                    target_view2=target_view2,
                    src_points_1=self.get_ref_points(ref_points_view1),
                    src_points_2=self.get_ref_points(ref_points_view2),
                    proj_map_1=proj_map_1,
                    proj_map_2=proj_map_2,
                )
                t1.append(frame_num)
                t2.append(frame_num)
                target_frame_num.append(t1)
                target_frame_num.append(t2)

        target_frame_num = np.array(target_frame_num)
        target = target_frame_num.T

        # If there is no bat, just add court points.
        if target.shape[-1] == 0:
            court = np.array(self.get_ref_points()).T
            target = None

        print("total target points: %d" % target.shape[-1])

        if show:
            v = Visual3D()
            v.show_3D(
                input_video_1,
                input_video_2,
                all_target_view1,
                all_target_view2,
                court,
                target,
                frame_skip,
                end_frame,
                is_set_lim=True,
                add_court=True,
                court_category="baseball_bat",
                save_name=output_fig,
            )
        court = np.array(court).T
        return target_frame_num, court

    def project_points(
        self,
        src_points_1,
        src_points_2,
        dst_points,
        dist,
        mtx,
    ):
        points_view1 = np.array([src_points_1]).astype("float32")
        points_view2 = np.array([src_points_2]).astype("float32")
        dst_points_pnp = np.array([dst_points]).astype("float32")

        retval1, rvec1, tvec1 = cv2.solvePnP(dst_points_pnp, points_view1, mtx, dist)
        r1, _ = cv2.Rodrigues(rvec1)
        retval2, rvec2, tvec2 = cv2.solvePnP(dst_points_pnp, points_view2, mtx, dist)
        r2, _ = cv2.Rodrigues(rvec2)

        proj_map_1 = np.matmul(mtx, np.concatenate((r1, tvec1), axis=1))
        proj_map_2 = np.matmul(mtx, np.concatenate((r2, tvec2), axis=1))

        return proj_map_1, proj_map_2

    def draw2court(
        self, target_view1, target_view2, src_points_1, src_points_2, proj_map_1, proj_map_2
    ):
        """
        detectron2
        target_view1 = [(x0, y0), (x1, y1)]
        target_view2 = [(x0, y0), (x1, y1)]

        yolo
        target_view1 = [468, 499, 20, 14]
        target_view2 = [1045, 444, 17, 15]

        view1ball = np.array([[468+20*0.5, 499+14*0.5]], dtype=np.float32)
        view2ball = np.array([[1045+17*0.5, 444+15*0.5]], dtype=np.float32) # read img
        """

        points_view1 = np.array(src_points_1).astype("float32")
        points_view2 = np.array(src_points_2).astype("float32")
        target_view1 = np.array(target_view1).astype("float32")
        target_view2 = np.array(target_view2).astype("float32")

        # read img
        points1 = np.concatenate((points_view1, target_view1), axis=0)
        points2 = np.concatenate((points_view2, target_view2), axis=0)
        # print(points1, points2)

        pts1 = np.transpose(points1)
        pts2 = np.transpose(points2)
        pts4D = cv2.triangulatePoints(proj_map_1, proj_map_2, pts1, pts2)

        pts4D = pts4D[:, :] / pts4D[-1, :]
        x, y, z, w = pts4D

        target1 = [x[-1], y[-1], z[-1]]
        target2 = [x[-2], y[-2], z[-2]]

        courtX = x[:-2]
        courtY = y[:-2]
        courtZ = z[:-2]
        court = [courtX, courtY, courtZ]
        return court, target1, target2


    def get_ref_points(self, ref_points="dst_points"):

        if ref_points == "dst_points":
            return [
                [0, 0, 0],
                [0, 0, 27],
                [0, 35, 27],
                [46, 0, 0],
                [46, 0, 27],
                [46, 35, 27],
            ]
        else:
            return np.load(ref_points)

