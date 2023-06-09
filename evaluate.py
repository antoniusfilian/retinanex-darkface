import numpy as np
import os

def compute_iou(box1, box2):
    # Menghitung Intersection over Union (IoU) antara dua kotak pembatas (boxes)
    #print ("box1: ", box1)
    #print ("box2: ", box2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = box1_area + box2_area - intersection_area

    if union == 0:
        iou = 0.0
    else:
        iou = intersection_area / union

    #print (box1_area)
    #print (box2_area)
    #print (intersection_area)
    #iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def compute_ap(gt_boxes, pred_boxes, iou_threshold=0.5):
    # Menghitung Average Precision (AP) berdasarkan prediksi dan ground truth boxes

    #print (gt_boxes)
    num_pred_boxes = len(pred_boxes)
    tp = np.zeros(num_pred_boxes)  # True Positives
    fp = np.zeros(num_pred_boxes)  # False Positives

    num_gt_boxes = len(gt_boxes)
    #print (num_gt_boxes)
    #print (num_pred_boxes)
    gt_matched = np.zeros(num_gt_boxes)  # Ground Truth boxes yang sudah di-matched

    # Mengurutkan prediksi berdasarkan skor (confidence score) secara menurun
    sorted_indices = np.argsort(pred_boxes[:, 4])[::-1]
    pred_boxes_sorted = pred_boxes[sorted_indices]

    # Iterasi melalui prediksi dan mencocokkan dengan ground truth boxes
    for i in range(num_pred_boxes):
        pred_box = pred_boxes_sorted[i]
        best_iou = -np.inf
        best_match_index = -1

        for j in range(num_gt_boxes):
            gt_box = gt_boxes[j]
            iou = compute_iou(pred_box[:4], gt_box)
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_match_index = j

        if best_iou > 0.0:
            if gt_matched[best_match_index] == 0:
                tp[i] = 1  # True Positive
                gt_matched[best_match_index] = 1
            else:
                fp[i] = 1  # False Positive
        else:
            fp[i] = 1  # False Positive

    # Menghitung Precision dan Recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / num_gt_boxes

    # Menghitung Average Precision (AP) menggunakan metode interpolasi titik
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11

    return ap

# Hitung AP
def compute_average_precision(gt_folder, pred_folder):
    # Menghitung nilai Average Precision (AP) dari folder ground truth dan folder prediksi
    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))
    num_files = min(len(gt_files), len(pred_files))

    #num_files = len(gt_files)
    #assert num_files == len(pred_files), "Jumlah file ground truth dan prediksi tidak sama."

    ap_total = 0.0
    for i in range(num_files):
        gt_file = os.path.join(gt_folder, gt_files[i])
        pred_file = os.path.join(pred_folder, pred_files[i])

        print (gt_file)
        print (pred_file)

        with open(gt_file, 'r') as ground_truth_file:
            gt_lines = ground_truth_file.readlines()

        with open(pred_file, 'r') as prediction_file:
            pred_lines = prediction_file.readlines()

        ground_truth_boxes = []
        for line in gt_lines:
            if line.startswith("#"):
                continue

            values = list(map(float, line.strip().split()))
            if len(values) == 4:
                x1min, y1min, x2max, y2max = values
                ground_truth_boxes.append([x1min, y1min, x2max, y2max])

        prediction_boxes = []
        for line in pred_lines:
            if line.startswith("#"):
                continue

            values = list(map(float, line.strip().split()))
            if len(values) == 5:
                x1min, y1min, x2max, y2max, confidence = values
                prediction_boxes.append([x1min, y1min, x2max, y2max, confidence])

        #gt_boxes = np.loadtxt(gt_file)
        #pred_boxes = np.loadtxt(pred_file)

        #print (np.loadtxt(gt_file))
        #print (ground_truth_boxes)

        gt_boxes = np.array(ground_truth_boxes, dtype = float)
        pred_boxes = np.array(prediction_boxes, dtype = float)

        # Hitung AP untuk setiap file
        ap = compute_ap(gt_boxes, pred_boxes)
        ap_total += ap

        # Tampilkan progress
        progress = (i + 1) / num_files * 100
        print("Progress: {:.2f}%".format(progress))

    # Hitung rata-rata AP
    ap_avg = ap_total / num_files
    return ap_avg

ground_truth_folder = "ground_truth/"
prediksi_folder = "predict/"
average_precision = compute_average_precision(ground_truth_folder, prediksi_folder)

print("Average Precision (AP):", average_precision)
