from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
img_path = '/02_tensorboard/dataset/train/ants_image/0013035.jpg'
image = Image.open(img_path)
img_array = np.array(image)

writer.add_image('test',img_array,1,dataformats="HWC")
for i in range(100):
    writer.add_scalar("y=x",i,i)

writer.flush()
writer.close()