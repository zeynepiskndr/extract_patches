#from read_image import PolygonDataset
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from extract_patches import extract_patches_2d
from UNet import UNetMini
import torch
from pathlib import Path

from PIL import Image
import cv2
import os
from torchsummary import summary
from tqdm import tqdm

# Use gpu for training if available else use cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 4
epochs = 4
learning_rate = 0.001

number_of_classes = 8

img_to_array = lambda image: np.asarray(image)
array_to_img = lambda array: Image.fromarray(array)

path = 'D:/Cyberneticlabs/zeynep'
image_path = "D:/Cyberneticlabs/zeynep/cropped_images/crop_images_new/"
mask_path = "D:/Cyberneticlabs/zeynep/cropped_images/mask_images_new/"
model_save_path = (path+"/weights/lines_segmentation.ckpt")
# define normalization transforms
transforms = transforms.Compose([

	transforms.ToPILImage(),
  # transforms.Resize([256, 256]),
	transforms.ToTensor()
])

def load_images_and_masks_in_path(images_path: Path, masks_path: Path):
    x = []
    y = []


    img_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)
    for image_file_name  in range(len(img_list)):

      image = cv2.imread(str(image_path)+str(img_list[image_file_name]))
      x.append(image)
    for mask_file_name  in range(len(mask_list)):
      mask = cv2.imread(str(mask_path)+str(mask_list[mask_file_name]))
      y.append(mask)
    # for image_file_name, mask_file_name in tqdm(zip(img_list, mask_list)):

    #   image = img_to_array(Image.open(image_path+image_file_name))
    #   mask = img_to_array(Image.open(mask_path+mask_file_name))

    #   x.append(image)
    #   y.append(mask)

    return np.array(x), np.array(y)



class FormsDataset(Dataset):

    def __init__(self, images, masks, num_classes: int, transforms=None):
        self.images = images
        self.masks = masks
        self.num_classes = num_classes
        self.transforms = transforms

    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.astype(np.float32)
        image = np.expand_dims(image, -1)
        image = image / 255

#         seed = random.randint(0, 2**31 - 1)
#         random.seed(seed) # apply this seed to img tranfsorms
#         torch.manual_seed(seed)
        if self.transforms:
            image = self.transforms(image)

        mask = self.masks[idx]
        mask = mask.astype(np.float32)
        mask = mask / 255
        mask[mask > .7] = 1
        mask[mask <= .7] = 0
#         mask = mask.astype(np.uint8)
#         mask = to_categorical(mask, self.num_classes).astype(np.int)

#         random.seed(seed) # apply this seed to target tranfsorms
#         torch.manual_seed(seed)
        if self.transforms:
            mask = self.transforms(mask)

        return image, mask

    def __len__(self):
        return len(self.images)


train_images, train_masks = load_images_and_masks_in_path(image_path, mask_path)
print(train_images, train_masks)
print(type(train_images))
print(type(train_masks))
train_dataset = FormsDataset(train_images, train_masks, number_of_classes, transforms(True))

image, mask = train_dataset[0]
image.shape, mask.shape

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for image, mask in train_data_loader:
    print(f"{image.shape}, {mask.shape}")
    print(f"{image.max()}, {mask.max()}")
    break

print(f'Train dataset has {len(train_data_loader)} batches of size {batch_size}')

model = UNetMini(number_of_classes).to(device)

summary(model, input_size=(1, 256, 256))  # (channels, H, W)

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_steps = len(train_data_loader)
print(f"{epochs} epochs, {total_steps} total_steps per epoch")

for epoch in range(epochs):
    for i, (images, masks) in enumerate(train_data_loader, 1):
        images = images.to(device)
        masks = masks.type(torch.LongTensor)
        masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        softmax = F.log_softmax(outputs, dim=1)
        loss = criterion(softmax, masks)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i) % 100 == 0:
            print (f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}")

torch.save(model.state_dict(), model_save_path)
print('FINISH')


