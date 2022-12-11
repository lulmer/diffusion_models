import gradio as gr
from ddpm import Diffusion
import torch
from modules import UNet_conditional
from utils import plot_images, get_data
from PIL import Image
import torchvision.transforms as tv 
import os 
import matplotlib.pyplot as plt
os.environ['NO_PROXY'] = "127.0.0.1"
img_size = 64
device='cuda'
PATH = os.path.join("models","airplanes_foolish_lion5","ckpt12000.pt")

diffusion = Diffusion(img_size=img_size, device=device)
model = UNet_conditional(image_size=img_size).to(device)
model.load_state_dict(torch.load(PATH))
model.eval()


def predict(resample_steps,source_img):
    mask = tv.functional.pil_to_tensor(source_img["mask"])
    mask= tv.functional.resize(mask, (64,64))
    print(type(source_img["mask"]))
    print(type(source_img["image"]))
    source_img["mask"].save("mask.png")
    source_img["image"].save("img.png")
    for i in range(0,4):
        plt.imsave(f"mask_no_{i}.png",mask[i,:,:])

    mask = mask[0,:,:]
    
    mask= mask/255
    print(mask)
    mask = torch.ones_like(mask)-mask
    print(torch.unique(mask))
    img= tv.functional.pil_to_tensor(source_img["image"])
    img= tv.functional.resize(img, (64,64))
    img= tv.functional.normalize(img/255,[0.5,0.5,0.5],[0.5,0.5,0.5])
    
    print(img.shape)
    print(mask.shape)
    diffusion.resample_steps= resample_steps
    sampled_images = diffusion.inpaint(model, n=1, x_0=img.to(device), mask=mask.to(device))
    print(sampled_images.shape)
    result = tv.functional.to_pil_image(sampled_images.squeeze(0).to("cpu"))
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
