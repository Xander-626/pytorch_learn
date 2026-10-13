from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img = Image.open("/Users/ripple/pyproject/pytorch-learning/02_tensorboard/dataset/train/ants_image/0013035.jpg")

mytfm = transforms.ToTensor()
img_tensor = mytfm(img)

my_writer = SummaryWriter("logs")
my_writer.add_image('Tensor_img',img_tensor)
my_writer.close()