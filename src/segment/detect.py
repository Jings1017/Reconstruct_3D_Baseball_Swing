from detectron2.structures import BoxMode
import detectron2
import numpy as np
import os, cv2, json
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
import time
from tqdm import tqdm, trange

setup_logger()

def get_bat_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        file_name = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(file_name).shape[:2]

        record["file_name"] = file_name
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for anno in annos:
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("bat_" + d, lambda d=d: get_bat_dicts("dataset/sony_1122/" + d))
    MetadataCatalog.get("bat_" + d).set(thing_classes=["baseball bat"])
bat_metadata = MetadataCatalog.get("bat_train")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = './out/model'
cfg.MODEL.DEVICE = 'cuda'
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.78   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


start_time = time.time()
coordinate = [] 
coordinate1 = []
coordinate2 = []

coordinate_bbox = []
curr_coordinate = [(0,0), (0,0)] # [head, tail]
curr_coordinate1 = [(0,0), (0,0)]
curr_coordinate2 = [(0,0), (0,0)]
curr_bbox_coordinate = [(0,0), (0,0)]

video_subpath = 'baseball_swing_20221122/view1'
video_name = '1_view1'
out_target_name = '1_view1_target.npy'

in_dir = os.path.join('./input', video_subpath)
out_dir = os.path.join('./out', video_subpath)

frame_folder = os.path.join(in_dir, video_name)
result_folder = os.path.join(out_dir, video_name)

if not os.path.exists(result_folder):
    os.makedirs(result_folder)
file_num = len(os.listdir(frame_folder))

output_folder = os.path.join('./out', video_subpath)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

target_path = os.path.join(output_folder, out_target_name)


for idx in trange(file_num-1):
    file_name = 'frame{}.png'.format(idx+1)
    file_path = os.path.join(frame_folder, file_name)
    mask_name = 'frame_{:04d}_mask.png'.format(idx)
    line_name = 'frame_{:04d}_line.png'.format(idx)

    mask_path = os.path.join(result_folder, mask_name)
    line_path = os.path.join(result_folder, line_name)

    im = cv2.imread(file_path)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                    metadata=bat_metadata, 
                    scale=1, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))  # 34: baseball bat
    boundingbox = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    classes = outputs["instances"].pred_classes.tolist()
    int_boundingbox = boundingbox.astype(int)

    x1, x2, y1, y2 = 0, 0, 0, 0
    if len(int_boundingbox)!=0:
        x1 = int_boundingbox[0][0]
        y1 = int_boundingbox[0][1]
        x2 = int_boundingbox[0][2]
        y2 = int_boundingbox[0][3]
    else:
        coordinate1.append(curr_coordinate1)
        coordinate2.append(curr_coordinate2)
        # coordinate_bbox.append(curr_bbox_coordinate)
        continue

    check_range = int(min((x2-x1), (y2-y1))/5)

    check_range = 10

    # curr_coordinate = []
    # ================================= find the shape of the bat =========================================== 

    ## for top left 
    count_top_left = 0
    for i in range(y1,y1+check_range):
        for j in range(x1, x1+check_range):
            if(outputs["instances"].pred_masks[0][i][j].item()):
                count_top_left +=1

    ## for bottom-left
    count_bottom_left = 0
    for i in range(y2-check_range,y2):
            for j in range(x1, x1+check_range):
                if(outputs["instances"].pred_masks[0][i][j].item()):
                    count_bottom_left +=1

    ## for top-right 
    count_top_right = 0
    for i in range(y1,y1+check_range):
        for j in range(x2-check_range, x2):
            if(outputs["instances"].pred_masks[0][i][j].item()):
                count_top_right +=1

    ## for bottom-right  
    count_bottom_right = 0
    for i in range(y2-check_range,y2):
        for j in range(x2-check_range, x2):
            if(outputs["instances"].pred_masks[0][i][j].item()):
                count_bottom_right +=1

    tmp_x1 = 0
    tmp_y1 = 0

    tmp_x2 = 0
    tmp_y2 = 0

    # -----------------------------------------------------------------
    # decide / or \
    mode = 0

    if count_top_left+count_bottom_right > count_bottom_left+count_top_right:
        tmp_x1 = x1
        tmp_y1 = y1
        tmp_x2 = x2
        tmp_y2 = y2
        mode = 1
        # curr_bbox_coordinate = [(tmp_x1,tmp_y1), (tmp_x2,tmp_y2)]
        # coordinate_bbox.append(curr_bbox_coordinate)
    else :
        tmp_x1 = x1
        tmp_y1 = y2
        tmp_x2 = x2
        tmp_y2 = y1
        mode = 2
        # curr_bbox_coordinate = [(tmp_x1,tmp_y1), (tmp_x2,tmp_y2)]
        # coordinate_bbox.append(curr_bbox_coordinate)

    diff_y = abs(tmp_y2-tmp_y1)
    diff_x = abs(tmp_x2-tmp_x1)

    min_x = min(tmp_x1, tmp_x2)
    min_y = min(tmp_y1, tmp_y2)

    #=================== mode 1 =================
    if mode == 1:
        # print('====== mode 1 =====')
        #---------------------------  top   seg ------------------------
        top_x1 = 0
        top_y1 = 0
        for y in range(diff_y):
            cnt = 0
            for x in range(diff_x):
                if outputs["instances"].pred_masks[0][y+min_y][x+min_x].item() == True:
                    cnt += 1
                    top_x1 = x+min_x
                    top_y1 = y+min_y
                    break
            if cnt>0:
                break

        top_x2 = 0
        top_y2 = 0

        for x in range(diff_x):
            cnt = 0
            for y in range(diff_y):
                if outputs["instances"].pred_masks[0][y+min_y][x+min_x].item() == True:
                    cnt += 1
                    top_x2 = x+min_x
                    top_y2 = y+min_y
                    break
            if cnt>0:
                break

        tmp_x1 = int((top_x1+top_x2)/2)
        tmp_y1 = int((top_y1+top_y2)/2)

        # -------------------------- bottom seg ------------------------
        bottom_x1 = 0
        bottom_y1 = 0
        for y in range(diff_y-1, 0, -1):
            cnt = 0
            for x in range(diff_x-1, 0, -1):
                if outputs["instances"].pred_masks[0][y+min_y][x+min_x].item() == True:
                    cnt += 1
                    bottom_x1 = x+min_x
                    bottom_y1 = y+min_y
                    break
            if cnt>0:
                break

        bottom_x2 = 0
        bottom_y2 = 0

        for x in range(diff_x-1, 0, -1):
            cnt = 0
            for y in range(diff_y-1, 0, -1):
                if outputs["instances"].pred_masks[0][y+min_y][x+min_x].item() == True:
                    cnt += 1
                    bottom_x2 = x+min_x
                    bottom_y2 = y+min_y
                    break
            if cnt>0:
                break
            
        tmp_x2 = int((bottom_x1+bottom_x2)/2)
        tmp_y2 = int((bottom_y1+bottom_y2)/2)

    #========================= mode 2 ===============

    elif mode == 2:

        # print('====== mode 2 =====')
        #---------------------------  top   seg ------------------------
        top_x1 = 0
        top_y1 = 0
        for y in range(diff_y-1,0, -1):
            cnt = 0
            for x in range(diff_x):
                if outputs["instances"].pred_masks[0][y+min_y][x+min_x].item() == True:
                    cnt += 1
                    top_x1 = x+min_x
                    top_y1 = y+min_y
                    break
            if cnt>0:
                break

        top_x2 = 0
        top_y2 = 0

        for x in range(diff_x):
            cnt = 0
            for y in range(diff_y-1, 0, -1):
                if outputs["instances"].pred_masks[0][y+min_y][x+min_x].item() == True:
                    cnt += 1
                    top_x2 = x+min_x
                    top_y2 = y+min_y
                    break
            if cnt>0:
                break

        tmp_x1 = int((top_x1+top_x2)/2)
        tmp_y1 = int((top_y1+top_y2)/2)

        # -------------------------- bottom seg ------------------------
        bottom_x1 = 0
        bottom_y1 = 0
        for y in range(diff_y):
            cnt = 0
            for x in range(diff_x-1, 0, -1):
                if outputs["instances"].pred_masks[0][y+min_y][x+min_x].item() == True:
                    cnt += 1
                    bottom_x1 = x+min_x
                    bottom_y1 = y+min_y
                    break
            if cnt>0:
                break

        bottom_x2 = 0
        bottom_y2 = 0

        for x in range(diff_x-1, 0, -1):
            cnt = 0
            for y in range(diff_y):
                if outputs["instances"].pred_masks[0][y+min_y][x+min_x].item() == True:
                    cnt += 1
                    bottom_x2 = x+min_x
                    bottom_y2 = y+min_y
                    break
            if cnt>0:
                break

        tmp_x2 = int((bottom_x1+bottom_x2)/2)
        tmp_y2 = int((bottom_y1+bottom_y2)/2)

        #-----------------------------------------------------------


    len_tb1 = pow((top_x1-bottom_x1),2)+pow((top_y1-bottom_y1),2)
    len_tb2 = pow((top_x2-bottom_x2),2)+pow((top_y2-bottom_y2),2)
    len_tb3 = pow((tmp_x1-tmp_x2),2)+pow((tmp_y1-tmp_y2),2)

    max_len = max(len_tb1, len_tb2, len_tb3)

    # calculate the distance

    t12 = pow((top_x1-top_x2),2)+pow((top_y1-top_y2),2)
    t1t = pow((top_x1-tmp_x1),2)+pow((top_y1-tmp_y1),2)
    t2t = pow((top_x2-tmp_x1),2)+pow((top_y2-tmp_y1),2)

    b12 = pow((bottom_x1-bottom_x2),2)+pow((bottom_y1-bottom_y2),2)
    b1t = pow((bottom_x1-tmp_x1),2)+pow((bottom_y1-tmp_y1),2)
    b2t = pow((bottom_x2-tmp_x1),2)+pow((bottom_y2-tmp_y1),2)

    if max_len == len_tb1:
        # print('len tb1')
        # head = pow((curr_coordinate[1][0]-top_x1),2)+pow((curr_coordinate[1][1]-top_y1),2)
        # tail = pow((curr_coordinate[1][0]-bottom_x1),2)+pow((curr_coordinate[1][1]-bottom_y1),2)
        # if head >= tail or idx<=100 :
        #     curr_coordinate = [(top_x1,top_y1), (bottom_x1,bottom_y1)]
        # else:
        #     curr_coordinate = [(bottom_x1,bottom_y1), (top_x1,top_y1)]
        curr_coordinate1 = [(top_x1,top_y1), (bottom_x1,bottom_y1)]
        curr_coordinate2 = [(bottom_x1,bottom_y1), (top_x1,top_y1)]
        cv2.circle(im, (top_x1,top_y1), 3, (0,255,0), 4)
        cv2.circle(im, (bottom_x1,bottom_y1), 3, (0,255,0), 4)
        cv2.line(im, (top_x1,top_y1), (bottom_x1,bottom_y1), (0,0,255), 5)
    elif max_len == len_tb2:
        # print('len tb2')
        # head = pow((curr_coordinate[1][0]-top_x2),2)+pow((curr_coordinate[1][1]-top_y2),2)
        # tail = pow((curr_coordinate[1][0]-bottom_x2),2)+pow((curr_coordinate[1][1]-bottom_y2),2)
        # if head >= tail or idx<=100 :
        #     curr_coordinate = [(top_x2,top_y2), (bottom_x2,bottom_y2)]
        # else:
        #     curr_coordinate = [(bottom_x2,bottom_y2), (top_x2,top_y2)]
        curr_coordinate1 = [(top_x2,top_y2), (bottom_x2,bottom_y2)]
        curr_coordinate2 = [(bottom_x2,bottom_y2), (top_x2,top_y2)]
        cv2.circle(im, (top_x2,top_y2), 3, (0,255,255), 4)
        cv2.circle(im, (bottom_x2,bottom_y2), 3, (0,255,255), 4)
        cv2.line(im, (top_x2,top_y2), (bottom_x2,bottom_y2), (0,0,255), 5)
    else:
        # head = pow((curr_coordinate[1][0]-tmp_x1),2)+pow((curr_coordinate[1][1]-tmp_y1),2)
        # tail = pow((curr_coordinate[1][0]-tmp_x2),2)+pow((curr_coordinate[1][1]-tmp_y2),2)
        # if head >= tail or idx<=100:
        #     curr_coordinate = [(tmp_x1,tmp_y1), (tmp_x2,tmp_y2)]
        # else:
        #     curr_coordinate = [(tmp_x2,tmp_y2), (tmp_x1,tmp_y1)]
        curr_coordinate1 = [(tmp_x1,tmp_y1), (tmp_x2,tmp_y2)]
        curr_coordinate2 = [(tmp_x2,tmp_y2), (tmp_x1,tmp_y1)]
        cv2.circle(im, (tmp_x1,tmp_y1), 3, (255,0,0), 4)
        cv2.circle(im, (tmp_x2,tmp_y2), 3, (255,0,0), 4)
        cv2.line(im, (tmp_x1,tmp_y1), (tmp_x2,tmp_y2), (0,0,255), 5)

    coordinate1.append(curr_coordinate1) 
    coordinate2.append(curr_coordinate2)

    cv2.imwrite(mask_path, out.get_image()[:,:,::-1])
    cv2.imwrite(line_path, im)



np.save(target_path, np.array(coordinate1))
npyfile = np.load(target_path, allow_pickle=True)
print('shape of target npy file  ', npyfile.shape)

end_time = time.time()

total_time = end_time - start_time
print('--------------------------------')
print('total time : ',total_time)
print('--------------------------------')

cv2.waitKey(0)
cv2.destroyAllWindows()