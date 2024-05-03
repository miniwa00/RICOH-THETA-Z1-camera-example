import requests
from requests.auth import HTTPDigestAuth
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os
# import utils
import sys

# transform = transforms.Compose(
#     [
#         # compressed image has to be converted to PIL to do resize
#         transforms.ToPILImage(),
#         transforms.Resize((640, 640)),
#         transforms.ToTensor(),
#     ]
# )
# device = torch.device("cpu")
# script_path = os.path.abspath(__file__)
# model_path = os.path.join(os.path.dirname(script_path), "model.torchscript")
# model = torch.jit.load(model_path, map_location=torch.device("cpu"))
# for name, param in model.named_parameters():
#     print(f"Parameter '{name}' on device: {param.device}")

url = "http://192.168.120.18/osc/commands/execute"
# url = "http://192.168.120.28/osc/commands/execute"
username = "THETAYR12100857"
password = "12100857"

# example information
# url = 'http://192.168.2.101/osc/commands/execute'
# username = "THETAYR14010001"
# password = "14010001"

# access point mode
# url = "http://192.168.120.28/osc/commands/execute"


payload = {"name": "camera.getLivePreview"}

headers = {"Content-Type": "application/json;charset=utf-8"}

# password only need for client mode
response = requests.post(
    url,
    auth=HTTPDigestAuth(username, password),
    json=payload,
    headers=headers,
    stream=True,
)
print(f'this is res : {response}')
# response = requests.post(url, json=payload, headers=headers, stream=True)


# showWindow 1: normal view
# showWindow 2: canny edge detection
def resize_bbox(bbox, original_size, current_size):
    """
    Resize bounding box coordinates from the current size to the original size.
    Args:
        bbox (torch.Tensor): Bounding box coordinates (xmin, ymin, xmax, ymax).
        original_size (tuple): Original image size (height, width).
        current_size (tuple): Current image size (height, width).
    Returns:
        torch.Tensor: Resized bounding box coordinates.
    """
    scale_factor_height = original_size[0] / current_size[0]
    scale_factor_width = original_size[1] / current_size[1]

    resized_bbox = torch.zeros_like(bbox)
    resized_bbox[:, 0] = bbox[:, 0] * scale_factor_width  # xmin
    resized_bbox[:, 1] = bbox[:, 1] * scale_factor_height  # ymin
    resized_bbox[:, 2] = bbox[:, 2] * scale_factor_width  # xmax
    resized_bbox[:, 3] = bbox[:, 3] * scale_factor_height  # ymax
    return resized_bbox


showWindow = 1

if response.status_code == 200:
    bytes_ = bytes()
    check = 0
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            bytes_ += chunk
            a = bytes_.find(b"\xff\xd8")
            b = bytes_.find(b"\xff\xd9")

            if a != -1 and b != -1:
                jpg = bytes_[a : b + 2]
                bytes_ = bytes_[b + 2 :]
                check += 1
                # print("before check : ", check)
                if check % 40 != 1:
                    continue
                # print("after check : ", check)

                img = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                # print(img.shape[:2])
                if showWindow == 1:
                    # img_tensor = transform(img)
                    # img_tensor = img_tensor.unsqueeze(0)
                    # img_tensor.to(torch.device("cpu"))

                    # model = torch.jit.load(
                    #     model_path, map_location=torch.device("cpu")
                    # )
                    # model.to(torch.device("cpu"))
                    # rst = model(img_tensor)
                    # bobs = utils.non_max_suppression(
                    #     prediction=rst[0],  # classes=[0, 1, 2, 3, 4]
                    # )
                    # resized_bboxes = resize_bbox(
                    #     bobs[0], img.shape[:2], (640, 640)
                    # )
                    # for bbox in resized_bboxes:
                    #     cv2.rectangle(
                    #         img,
                    #         (int(bbox[0]), int(bbox[1])),
                    #         (int(bbox[2]), int(bbox[3])),
                    #         (0, 255, 0),
                    #         2,
                    #     )

                    # sys.exit(1)
                    # print(rst)
                    cv2.imshow("Preview", img)

                if showWindow == 2:
                    # Convert to graycsale
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Blur the image for better edge detection
                    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
                    # Canny Edge Detection
                    edges = cv2.Canny(
                        image=img_blur, threshold1=100, threshold2=200
                    )  # Canny Edge Detection
                    # Display Canny Edge Detection Image
                    cv2.imshow("Canny Edge Detection", edges)

                # ESC key will quit
                if cv2.waitKey(1) == 27:
                    break


else:
    print("Error: ", response.status_code)

cv2.destroyAllWindows()
