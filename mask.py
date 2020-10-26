import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def mask_feat(feat, mask, s=1):
    mask_feat = np.zeros_like(feat)
    r = int(mask.shape[0] / 2)
    h, w = feat.shape

    for i in range(0, h, s):
        for j in range(0, w, s):
            x_min = max(0, i-r)
            x_max = min(h-1, i+r)
            y_min = max(0, j-r)
            y_max = min(w-1, j+r)
            m_x_min = 0 if i-r>=0 else r-i
            m_x_max = mask.shape[0] if i+r<h else mask.shape[0]-(i+r-h+1)
            m_y_min = 0 if j-r>=0 else r-j
            m_y_max = mask.shape[1] if j+r<w else mask.shape[1]-(j+r-w+1)
            mask_feat[x_min:x_max+1, y_min:y_max+1] += feat[i, j] * mask[m_x_min:m_x_max, m_y_min:m_y_max]
    return mask_feat+feat

if __name__ == "__main__":
    T = 8
    np.set_printoptions(precision=3)
    mask_list = {
        "center": np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]]),
        "top": np.array([[1, 1, 1],
                         [1, 1, 1],
                         [0, 0, 0]]),
        "bottom": np.array([[0, 0, 0],
                         [1, 1, 1],
                         [1, 1, 1]]),
        "left": np.array([[1, 1, 0],
                         [1, 1, 0],
                         [1, 1, 0]]),
        "right": np.array([[0, 1, 1],
                         [0, 1, 1],
                         [0, 1, 1]]),
        "top left": np.array([[1, 1, 0],
                         [1, 1, 0],
                         [0, 0, 0]]),
        "top right": np.array([[0, 1, 1],
                         [0, 1, 1],
                         [0, 0, 0]]),
        "bottom right": np.array([[0, 0, 0],
                         [0, 1, 1],
                         [0, 1, 1]]),
        "bottom left": np.array([[0, 0, 0],
                         [1, 1, 0],
                         [1, 1, 0]]),

    }
    feat = np.zeros((13, 13))
    # get object center point and responding feature map
    center = [[5,3], [9,8]]
    for c  in center:
        feat[c[0], c[1]] = 1

    for k in mask_list:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        mask = mask_list[k] / 2.
        h, w = mask.shape
        mask = F.softmax(torch.from_numpy(mask).reshape(-1)).reshape(h, w).numpy()
        m_feat = feat.copy()
        for i in range(T):
            m_feat = mask_feat(m_feat, mask)
        
#        feat = feat / feat.max()
#        m_feat = m_feat / m_feat.max()
        ax1.imshow(feat, cmap='coolwarm', interpolation='bilinear')
        ax1.set_title('Before')
#        ax1.grid(True)
        ax2.imshow(m_feat, cmap='coolwarm', interpolation='bilinear')
#        ax2.grid(True)
        ax2.set_title('After {}'.format(k))
        plt.show()
        

