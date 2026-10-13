from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import __future__
img_path = '/Users/ripple/pyproject/pytorch-learning/02_tensorboard/dataset/train/ants_image/0013035.jpg'
img = Image.open(img_path)

#totensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
#normalize
trans_nomalize = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_nomalize(img_tensor)

#resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_totensor(trans_resize(img))
#compose_resize 2
trans_resize_2 = transforms.Resize(428)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_compose = trans_compose(img)
#randomcrop
rmcrop = transforms.Compose([transforms.RandomCrop((256,512)),transforms.ToTensor()])

my_writer = SummaryWriter("logs")
my_writer.add_image('Tensor_img',img_tensor)
my_writer.add_image('Norm',img_norm)
my_writer.add_image('Resize',img_resize)
my_writer.add_image('Compose',img_compose)

for i in range(10):
    img_rmcrop = rmcrop(img)
    my_writer.add_image('rmcrop',img_rmcrop,i)
my_writer.close()
