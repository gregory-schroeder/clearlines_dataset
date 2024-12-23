from pathlib import Path
import cv2
import numpy as np
import random
import argparse


def draw_edge_segments_connected(img, edge_segments, color=None):
    img_edges_on_original = np.copy(img)

    if len(img_edges_on_original.shape) == 2:  # Check if the image is grayscale
        img_edges_on_original = cv2.cvtColor(img_edges_on_original, cv2.COLOR_GRAY2BGR)

    b_random_color = False
    if color is None:
        b_random_color = True

    for idx_edge_seg, edge_segment in enumerate(edge_segments):
        if b_random_color:
            B = random.randint(80, 255)
            G = random.randint(80, 255)
            R = random.randint(80, 255)
            color = (B, G, R)

        edge_segment = edge_segment.astype(np.int32)
        edge_segment = edge_segment.reshape((-1, 1, 2))
        img_edges_on_original = cv2.polylines(img_edges_on_original, [edge_segment], isClosed=False, color=color, thickness=2)

        # print idx of edge-segment
        x_min, x_max, y_min, y_max = min(edge_segment[:, 0, 0]), max(edge_segment[:, 0, 0]), min(edge_segment[:, 0, 1]), max(edge_segment[:, 0, 1])
        w, h = x_max - x_min, y_max - y_min
        edge_extent = max((w, h))

        text = str(idx_edge_seg) # + ': ' + str(int(edge_extent))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2 # 0.4
        x, y = int(edge_segment[0, 0, 0]), int(edge_segment[0, 0, 1])
        cv2.putText(img_edges_on_original, text, (x, y), font, font_scale, color, thickness=1, lineType=cv2.LINE_AA)

    return img_edges_on_original


def pad_coordinates(x1, y1, x2, y2, min_size):
    if x2 - x1 < min_size:
        pad = (min_size - (x2 - x1)) / 2
        x1 = x1 - pad
        x2 = x2 + pad
    if y2 - y1 < min_size:
        pad = (min_size - (y2 - y1)) / 2
        y1 = y1 - pad
        y2 = y2 + pad
    return x1, y1, x2, y2

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1_gt, y1_gt, x2_gt, y2_gt = box1[:,0,0].min(), box1[:,0,1].min(), box1[:,0,0].max(), box1[:,0,1].max()
    x1_est, y1_est, x2_est, y2_est = box2[:,0,0].min(), box2[:,0,1].min(), box2[:,0,0].max(), box2[:,0,1].max()

    # require a minimum height/width of 10 pixels to calculate reasonable iou
    min_width_height = 10
    x1_gt, y1_gt, x2_gt, y2_gt = pad_coordinates(x1_gt, y1_gt, x2_gt, y2_gt, min_width_height)
    x1_est, y1_est, x2_est, y2_est = pad_coordinates(x1_est, y1_est, x2_est, y2_est, min_width_height)

    # Compute intersection
    xi1 = max(x1_gt, x1_est)
    yi1 = max(y1_gt, y1_est)
    xi2 = min(x2_gt, x2_est)
    yi2 = min(y2_gt, y2_est)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Compute union
    box1_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    box2_area = (x2_est - x1_est) * (y2_est - y1_est)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def precision_recall_from_boxes(gt_boxes, pred_boxes, iou_threshold):
    """Calculate precision and recall based on IoU matches."""
    matches = []
    for pred_box in pred_boxes:
        ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
        # print('ious=', ious)
        if max(ious) >= iou_threshold:
            matches.append(1)  # True Positive
        else:
            matches.append(0)  # False Positive

    tp = sum(matches)
    fp = len(matches) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall



def calculate_tp_fp_fn(gt_boxes, pred_boxes, iou_threshold):
    tp, fp, fn = [], [], []
    matched_gt = set()  # Track matched ground truth boxes

    # Iterate over predicted boxes
    for pred_box in pred_boxes:
        ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
        max_iou = max(ious) if ious else 0
        if max_iou >= iou_threshold:
            matched_idx = ious.index(max_iou)
            if matched_idx not in matched_gt:
                tp.append(pred_box)  # True Positive
                matched_gt.add(matched_idx)
            else:
                fp.append(pred_box)  # False Positive (duplicate match)
        else:
            fp.append(pred_box)  # False Positive

    # Collect unmatched ground truth boxes (False Negatives)
    for idx, gt_box in enumerate(gt_boxes):
        if idx not in matched_gt:
            fn.append(gt_box)

    return tp, fp, fn

def visualize_est_vs_gt(img, edge_segments_gt, edge_segments_est, iou_threshold):
    if len(img.shape) == 3:  # Check if the image is color and convert to grayscale for better visualization
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tp, fp, fn = calculate_tp_fp_fn(edge_segments_gt, edge_segments_est, iou_threshold)
    img = draw_edge_segments_connected(img, tp, color=(0, 255, 0))
    img = draw_edge_segments_connected(img, fp, color=(0, 0, 255))
    img = draw_edge_segments_connected(img, fn, color=(0, 165, 255))
    return img


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate edge-segment detection.")
    parser.add_argument('--ground-truth', required=True, type=str, help="Path to the directory containing ground truth labels (.npy files).")
    parser.add_argument('--predictions', required=True, type=str, help="Path to the directory containing predicted edge segments (.npy files).")
    parser.add_argument('--images', type=str, help="Path to the directory containing input images for visualization (optional).")
    parser.add_argument('--output', type=str, default="evaluation/visualization", help="Path to save the visualization results.")
    args = parser.parse_args()

    # Parse directories
    edges_dir_gt    = Path(args.ground_truth)
    edges_dir_est   = Path(args.predictions)
    results_dir     = Path(args.output)
    img_dir         = Path(args.images) if args.images else None # optional

    if img_dir is not None:
        img_paths = sorted(img_dir.glob('*.png'))
        assert len(img_paths) > 0, 'No images for visualization found. Set img_dir = None, if no visualization is required.'
    edges_paths_gt = sorted(edges_dir_gt.glob('*.npy'))
    edges_paths_est= sorted(edges_dir_est.glob('*.npy'))
    assert len(edges_paths_gt) > 0, 'No gt found'
    assert len(edges_paths_gt) == len(edges_paths_est), 'Unequal number of ground-truth and estimated files found'

    iou_threshold = 0.5

    precisions_per_image, recalls_per_image = [], []
    for idx_file in range(len(edges_paths_gt)):
        edge_segments_gt = list(np.load(edges_paths_gt[idx_file], allow_pickle=True))
        edge_segments_est = list(np.load(edges_paths_est[idx_file], allow_pickle=True))

        precision, recall = precision_recall_from_boxes(edge_segments_gt, edge_segments_est, iou_threshold)
        # print('precision, recall = ', precision, recall)
        precisions_per_image.append(precision)
        recalls_per_image.append(recall)

        if img_dir is not None:
            img = cv2.imread(img_paths[idx_file])
            img_vis_tp_fp_fn = visualize_est_vs_gt(img, edge_segments_gt, edge_segments_est, iou_threshold)
            img_vis_tp_fp_fn = cv2.resize(img_vis_tp_fp_fn, (0, 0), fx=0.5, fy=0.5)
            # cv2.imshow('Visualization tp, fp and fn', img_vis_tp_fp_fn)
            # cv2.waitKey()
            save_path_img_vis = results_dir / img_paths[idx_file].name
            cv2.imwrite(save_path_img_vis, img_vis_tp_fp_fn)

    average_precision = sum(precisions_per_image) / len(precisions_per_image)
    average_recall = sum(recalls_per_image) / len(recalls_per_image)

    F1_score_iou05 = 2*(average_precision * average_recall) / (average_precision + average_recall)
    print(f"average_precision: {average_precision:.2f}")
    print(f"average_recall: {average_recall:.2f}")
    print(f"F1_score: {F1_score_iou05:.2f}")

    # Write results to the summary file
    savepath_summary = results_dir.parent / 'summary.txt'
    with open(savepath_summary, 'w') as summary_file:
        summary_file.write(f"Results for Intersection over Union (IoU) of: {iou_threshold}\n")
        summary_file.write(f"Average Precision: {average_precision:.2f}\n")
        summary_file.write(f"Average Recall: {average_recall:.2f}\n")
        summary_file.write(f"F1 Score;: {F1_score_iou05:.2f}\n")

    print(f"Summary saved to: {savepath_summary}")

