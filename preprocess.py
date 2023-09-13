import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def img_convert(input: np.ndarray) -> torch.Tensor:
    """
    input:
    - type is np.ndarray.
    - shape: (H, W, C), where C (Channel) is BGR (Blue, Green, Red) type.

    output:
    - type is torch.Tensor.
    - shape: (C, H, W) where C (Channel) is RGB (Red, Green, Blue) type.
    """ 
    
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    input = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)


    # input = cv2.medianBlur(input, 5)   # 模糊化
    input = cv2.GaussianBlur(input,(5, 5), 0)

    _, input = cv2.threshold(input, 92, 255, cv2.THRESH_BINARY)

    input =cv2.equalizeHist(input)


    input = cv2.resize(input, (64, 64))

    cv2.imwrite('image1.png',input)

    # for i in range(len(input)):
    #     for j in range(len(input[i])):
    #         input[i][j] = abs(255 - input[i][j])

    min_val = np.min(input)
    max_val = np.max(input)
    output = (input - min_val) / (max_val - min_val)

    output = torch.from_numpy(output)
    output = output.to(torch.float32)

    output = output.unsqueeze(0)

    

    return output
