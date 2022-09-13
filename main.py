#from read_image import PolygonDataset
from read_json import Test

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from extract_patches import extract_patches_2d
from UNet import UNetMini

# define normalization transforms
transforms = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, trainImages, trainBBoxes, transforms):
      # store the image and mask filepaths, and augmentation
      # transforms
      print('myDataset')
      self.imagePaths = trainImages
      self.bbox = trainBBoxes
      self.transforms = transforms
    def __len__(self):
      # return the number of total samples contained in the dataset
      print(len(self.imagePaths))
      return len(self.imagePaths)

    def __getitem__(self, idx):
    # grab the image path from the current index
        image = self.imagePaths[idx]

        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)

        bbox = self.bbox[idx]
        patches = extract_patches_2d(image, bbox)
        print(patches)

        print(image)
        print(bbox)       
            
        return image, bbox

        
NUM_EPOCHS = 40

trainImage, trainBBoxe = Test.main()
#print(trainImage, trainBBoxe)
#trainImage, trainBBoxe = self.image.read_image(image_id, list)
#patches = extract_patches_2d(trainImage, trainBBoxe)

trainDS = MyDataset(trainImage, trainBBoxe, transforms=transforms)
image, box = trainDS.__getitem__(0)
print(image)
print(box)

print(trainDS)
#print(trainDS[0])

trainLoader = DataLoader(trainDS, batch_size=32,shuffle=True)
print(trainLoader)

model = UNetMini

print('FINISH')
        

