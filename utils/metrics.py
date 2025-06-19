import torch

def calculate_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of 3D boxes
    Args:
        boxes1: (N, 7) [x, y, z, l, w, h, ry]
        boxes2: (M, 7) [x, y, z, l, w, h, ry]
    Returns:
        iou: (N, M)
    """
    from .bbox_utils import boxes_to_corners_3d
    
    corners1 = boxes_to_corners_3d(boxes1)  # (N, 8, 3)
    corners2 = boxes_to_corners_3d(boxes2)  # (M, 8, 3)
    
    # Compute min and max for each dimension
    min1 = corners1.min(dim=1)[0]  # (N, 3)
    max1 = corners1.max(dim=1)[0]  # (N, 3)
    
    min2 = corners2.min(dim=1)[0]  # (M, 3)
    max2 = corners2.max(dim=1)[0]  # (M, 3)
    
    # Expand dimensions for broadcasting
    min1_exp = min1.unsqueeze(1)  # (N, 1, 3)
    max1_exp = max1.unsqueeze(1)  # (N, 1, 3)
    min2_exp = min2.unsqueeze(0)  # (1, M, 3)
    max2_exp = max2.unsqueeze(0)  # (1, M, 3)
    
    # Compute intersection
    intersect_min = torch.max(min1_exp, min2_exp)  # (N, M, 3)
    intersect_max = torch.min(max1_exp, max2_exp)  # (N, M, 3)
    intersect_whd = torch.clamp(intersect_max - intersect_min, min=0)
    intersect_vol = intersect_whd[..., 0] * intersect_whd[..., 1] * intersect_whd[..., 2]
    
    # Compute union
    vol1 = (max1 - min1).prod(dim=1)  # (N,)
    vol2 = (max2 - min2).prod(dim=1)  # (M,)
    vol1_exp = vol1.unsqueeze(1)  # (N, 1)
    vol2_exp = vol2.unsqueeze(0)  # (1, M)
    union_vol = vol1_exp + vol2_exp - intersect_vol
    
    # Compute IoU
    iou = intersect_vol / (union_vol + 1e-6)
    return iou

def calculate_map(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP)
    Args:
        pred_boxes: List of (N, 7) predicted boxes for each sample
        pred_scores: List of (N,) predicted scores for each sample
        pred_labels: List of (N,) predicted labels for each sample
        gt_boxes: List of (M, 7) ground truth boxes for each sample
        gt_labels: List of (M,) ground truth labels for each sample
        iou_threshold: IoU threshold for positive detection
    Returns:
        mAP: mean Average Precision
    """
    aps = []
    for cls_idx in range(len(pred_labels[0].unique())):
        # Collect all predictions and GTs for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        
        for p_boxes, p_scores, p_labels, g_boxes, g_labels in zip(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
        ):
            # Filter predictions for this class
            cls_mask = (p_labels == cls_idx)
            if cls_mask.any():
                all_pred_boxes.append(p_boxes[cls_mask])
                all_pred_scores.append(p_scores[cls_mask])
            
            # Filter GTs for this class
            gt_mask = (g_labels == cls_idx)
            if gt_mask.any():
                all_gt_boxes.append(g_boxes[gt_mask])
        
        if not all_pred_boxes or not all_gt_boxes:
            continue
            
        # Sort predictions by score
        pred_boxes_cls = torch.cat(all_pred_boxes, dim=0)
        pred_scores_cls = torch.cat(all_pred_scores, dim=0)
        sorted_indices = torch.argsort(pred_scores_cls, descending=True)
        pred_boxes_cls = pred_boxes_cls[sorted_indices]
        
        # Calculate precision-recall curve
        tp = torch.zeros(len(pred_boxes_cls))
        fp = torch.zeros(len(pred_boxes_cls))
        gt_matched = [torch.zeros(len(gt)) for gt in all_gt_boxes]
        
        for i, pred_box in enumerate(pred_boxes_cls):
            max_iou = 0
            best_gt_idx = -1
            best_sample_idx = -1
            
            # Find best matching GT across all samples
            for sample_idx, gt_boxes_sample in enumerate(all_gt_boxes):
                if len(gt_boxes_sample) == 0:
                    continue
                    
                ious = calculate_iou(pred_box.unsqueeze(0), gt_boxes_sample)
                current_max_iou, current_gt_idx = ious.max(dim=1)
                
                if current_max_iou > max_iou:
                    max_iou = current_max_iou
                    best_gt_idx = current_gt_idx.item()
                    best_sample_idx = sample_idx
            
            if max_iou > iou_threshold and not gt_matched[best_sample_idx][best_gt_idx]:
                tp[i] = 1
                gt_matched[best_sample_idx][best_gt_idx] = 1
            else:
                fp[i] = 1
        
        # Compute precision-recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        recalls = tp_cumsum / (sum(len(gt) for gt in all_gt_boxes) + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Compute AP
        precisions = torch.cat([torch.tensor([1.]), precisions])
        recalls = torch.cat([torch.tensor([0.]), recalls])
        ap = torch.trapz(precisions, recalls)
        aps.append(ap)
    
    mAP = torch.mean(torch.tensor(aps)) if aps else torch.tensor(0.)
    return mAP
