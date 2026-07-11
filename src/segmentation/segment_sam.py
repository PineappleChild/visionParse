import cv2 

import numpy as np

def segmenting_with_sam(
                        original_image: np.ndarray,
                        targets: list,
                        sam_model
                        ):
    
    target_mask = np.zeros_like(np.asarray(targets["clusters"][0][0]), dtype=np.uint8)
    for cluster in targets["clusters"]:
        target_mask |= cluster[0].astype(np.uint8)

    kernel = np.ones((7,7), np.uint8)
    clean = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)

    total_area = target_mask.size
    min_area = max(50, int(0.0025 * total_area))
    sam_model.set_image(original_image)
    final_mask = np.zeros(target_mask.shape, dtype=bool)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        comp_mask = (labels == label_id).astype(np.uint8)
        x, y, w, h, _ = stats[label_id]

        pad = int(0.1 * max(w, h))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(target_mask.shape[1] - 1, x + w + pad)
        y1 = min(target_mask.shape[0] - 1, y + h + pad)
        box = np.array([x0, y0, x1, y1])

        #pos points 
        dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        ys, xs = np.where(comp_mask > 0)
        dvals = dist[ys, xs]
        order = np.argsort(-dvals)
        n_pos = min(5, len(order))
        top = order[:n_pos]
        pos_points = np.stack([xs[top], ys[top]], axis=1)

        #neg points
        #gotta change this, sample neg points from the background classified clusters, rn its trying to get something outside
        #of the target clusters, might be causing the problem with a large target getting segmented rather than a smaller background
        comp_dilated = cv2.dilate(comp_mask, kernel, iterations=3)
        local_bg = (1 - comp_dilated[y0:y1, x0:x1])
        bg_ys, bg_xs = np.where(local_bg)
        n_neg = min(5, len(bg_ys))

        if n_neg > 0:
            idxs = np.random.choice(len(bg_ys), size=n_neg, replace=False)
            neg_points = np.stack([bg_xs[idxs] + x0, bg_ys[idxs] + y0], axis=1)
        else:
            neg_points = np.empty((0, 2), dtype=int)

        coords = np.vstack([pos_points, neg_points]) if len(neg_points) > 0 else pos_points
        
        point_labels = np.concatenate([
            np.ones(len(pos_points), dtype=np.int32),
            np.zeros(len(neg_points), dtype=np.int32)
        ])

        masks, scores, logits = sam_model.predict(
            point_coords=coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )

        best_idx = np.argmax(scores)

        masks2, scores2, _ = sam_model.predict(
            point_coords=coords,
            point_labels=point_labels,
            box=box,
            mask_input=logits[best_idx][None, :, :],
            multimask_output=False,
        )

        best_mask = masks2[np.argmax(scores2)]

        comp_area = comp_mask.sum()
        mask_area = best_mask.sum()

        print(f"  component {label_id}: comp_area={comp_area} mask_area={mask_area} "
                f"ratio={mask_area / max(comp_area, 1):.2f}")

        ratio = mask_area / max(comp_area, 1)
        if 0.2 <= ratio <= 3.0:   
            final_mask |= best_mask

    return final_mask

        