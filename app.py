import gradio as gr
from ddpm import Diffusion
import torch
from modules import UNet_conditional
from utils import plot_images, get_data
from PIL import Image, ImageDraw
import torchvision.transforms as tv 
import os 
import matplotlib.pyplot as plt
import yaml
import numpy as np
import cv2 



os.environ['NO_PROXY'] = "127.0.0.1"
img_size = 64
device='cuda'
PATH = os.path.join("models","airplanes_foolish_lion5","ckpt12000.pt")

diffusion = Diffusion(img_size=img_size, device=device)
model = UNet_conditional(image_size=img_size).to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

def get_bounding_boxes(mask, img_size=64):
    """
    Given a binary mask, returns the bounding boxes coordinates of every continuous non-zero pixels on the mask.

    Parameters
    ----------
    mask: numpy.ndarray
        A binary mask of shape (H, W) where H is the height and W is the width of the mask.
    img_size : int
        size of the bounding box

    Returns
    -------
    List[Tuple[int, int, int, int]]
        A list of tuples, where each tuple represents the coordinates of a bounding box in the form (x1, y1, x2, y2) where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the bounding box

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.zeros((100,100))
    >>> mask[20:40, 20:40] = 1
    >>> bounding_boxes = get_bounding_boxes(mask, img_size=20)
    >>> bounding_boxes
    [(20, 20, 40, 40)]

    """
    # Find the indices of the non-zero elements in the mask
    # Convert the PyTorch tensor to a numpy array
    mask_np = mask.numpy().astype(np.uint8)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(1-mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Iterate through the contours and find the bounding box of each one
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = x + w//2, y + h//2
        
        # The bounding box is not too big
        # The bounding box is too big for the inpainting model
        # So we will create a new bounding box that is smaller
        new_x, new_y = center_x-(img_size//2), center_y-(img_size//2)
        
        bounding_boxes.append((new_x, new_y, new_x + img_size, new_y + img_size))
    pil_mask = tv.functional.to_pil_image(mask)
    for i,bbox in enumerate(bounding_boxes):
        
        draw = ImageDraw.Draw(pil_mask)
        draw.rectangle(bbox, outline="red")
        pil_mask.save(f"debug_imgs/bbox{i}.png")
    return bounding_boxes

def denormalize(x):
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x 



def inpaint_bboxes(img, bboxes, mask, diffusion_model,resample_steps):
    """
    Given an image, a list of bounding boxes and a mask, it inpaints the bounding boxes on the image using the given diffusion model.
    
    Parameters
    ----------
    img : numpy.ndarray
        Image of shape (C, H, W) where C is the number of channels, H is the height and W is the width of the image.
    bboxes : List[Tuple[int, int, int, int]]
        A list of tuples, where each tuple represents the coordinates of a bounding box in the form (x1, y1, x2, y2) where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the bounding box.
    mask : numpy.ndarray
        A binary mask of shape (H, W) where H is the height and W is the width of the mask.
    diffusion_model : Pytorch model
        A diffusion model that takes image and mask as input and returns the inpainted image.

    Returns
    -------
    PIL.Image
        An inpainted image

    Examples
    --------
    >>> from PIL import Image
    >>> img = Image.open("image.jpg")
    >>> img_np = np.array(img)
    >>> mask = np.zeros((100,100))
    >>> mask[20:40, 20:40] = 1
    >>> bboxes = get_bounding_boxes(mask, 20)
    >>> diffusion_model = diffusion_model.to(device)
    >>> inpainted_img = inpaint_bboxes(img_np, bboxes, mask, diffusion_model)
    >>> inpainted_img
    <PIL.Image.Image image mode=RGB size=100x100 at 0x7F0D0F8F5E50>
    """
    mean,std =[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]
    for bbox in bboxes:
        # Extract submask and subimage from the bounding box
        n,p = mask.shape
        submask = mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        subimage = img[:,bbox[1]:bbox[3],bbox[0]:bbox[2]]
        tv.functional.to_pil_image(mask).save('debug_imgs/mask.png')
        tv.functional.to_pil_image(img.squeeze(0)).save('debug_imgs/img.png')
        tv.functional.to_pil_image(submask).save('debug_imgs/submask.png')
        tv.functional.to_pil_image(subimage.squeeze(0)).save('debug_imgs/subimage.png')
        # Inpaint the subimage using the diffusion model
        diffusion.resample_steps= resample_steps
        sampled_image = diffusion.inpaint(diffusion_model, n=1, x_0=subimage.unsqueeze(0).to(device), mask=submask.to(device))
        tv.functional.to_pil_image(sampled_image.squeeze(0).to("cpu")).save('debug_imgs/sampled_subimage.png')
        # Replace the subimage with the inpainted image
        img = denormalize(img)
        img[:,bbox[1]:bbox[3],bbox[0]:bbox[2]] = sampled_image.squeeze(0).to("cpu")
    result = tv.functional.to_pil_image(img.squeeze(0).to("cpu"))
    result.save('debug_imgs/result.png')    
    # Convert the image back to a PIL image and return it
    return result


def predict(resample_steps,source_img):
    do_resize = False 
    mask = tv.functional.pil_to_tensor(source_img["mask"])
    mask = mask[0,:,:]
    if mask.shape[0] < img_size or mask.shape[1] < img_size :
        do_resize = True 
        mask= tv.functional.resize(mask, (img_size,img_size))

    print(type(source_img["mask"]))
    print(type(source_img["image"]))
    source_img["mask"].save("mask.png")
    source_img["image"].save("img.png")
    #for i in range(0,4):
    #    plt.imsave(f"mask_no_{i}.png",mask[i,:,:])
    mask= mask/255
    mask = torch.ones_like(mask)-mask

    bboxes = get_bounding_boxes(mask,img_size)
    img= tv.functional.pil_to_tensor(source_img["image"])
    if do_resize: 
        img= tv.functional.resize(img, (img_size,img_size))
    img= tv.functional.normalize(img/255,[0.5,0.5,0.5],[0.5,0.5,0.5])

    if do_resize:
        diffusion.resample_steps= resample_steps
        sampled_images = diffusion.inpaint(model, n=1, x_0=img.to(device), mask=mask.to(device))

        result = tv.functional.to_pil_image(sampled_images.squeeze(0).to("cpu"))
    else:
        result = inpaint_bboxes(img, bboxes, mask, model, resample_steps)
    
    result.save("result.png")
    return result


demo = gr.Interface(
    fn=predict,
    inputs = [
                gr.Slider(1, 25, value=4, step=1, label="Resampling Steps"),
                gr.Image(source="upload", type="pil", tool="sketch", elem_id="source_container"),
            ],
    outputs="image")

demo.launch()   
