import numpy as np
from numba import jit


@jit(nopython=True)
def compute_overlap(boxes, query_boxes):
    n = boxes.shape[0]
    k = query_boxes.shape[0]

    overlaps = np.zeros((n, k), dtype=np.float32)

    # for all anchors
    for x in range(k):
        box_area = (
                (query_boxes[x, 2] - query_boxes[x, 0] + 1) *
                (query_boxes[x, 3] - query_boxes[x, 1] + 1)
        )
        # for all gt boxes
        for i in range(n):
            iw = (
                    min(boxes[i, 2], query_boxes[x, 2]) -
                    max(boxes[i, 0], query_boxes[x, 0]) + 1
            )

            if iw > 0:
                ih = (
                    min(boxes[i, 3], query_boxes[x, 3]) -
                    max(boxes[i, 1], query_boxes[x, 1]) + 1
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[i, 2] - boxes[i, 0] + 1) *
                        (boxes[i, 3] - boxes[i, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[i, x] = iw * ih / ua

    return overlaps