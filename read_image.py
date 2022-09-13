import os
import cv2
import numpy as np
from PIL import Image

from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt
from PIL import Image, ImageChops, ImageOps
from io import BytesIO 

#you should give your local path
train_path = 'D:/Cyberneticlabs/CoWork/train/'
cropped_images_path = 'D:/Cyberneticlabs/CoWork/cropped_images/'

#transformations
train_transform = transforms.Compose([transforms.Grayscale(),
                                      transforms.ToTensor(),
                                      transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.5,1))
                                      ])

#remove border code
def trim(im):
  bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
  diff = ImageChops.difference(im, bg)
  diff = ImageChops.add(diff, diff, 2.0, -100)
  bbox = diff.getbbox()
  if bbox:
    return im.crop(bbox)

# Rotate the image around its center
def rotateImage(img, angle: float):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

#image preprocessing and find angle
def image_cleaning(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray_image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
  
  (_, thresh1) = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)

  blur3 = cv2.GaussianBlur(thresh1,(3,3),0)
  blurred = cv2.GaussianBlur(gray, (3, 3), 0)

  #thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
  thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
  dilate = cv2.dilate(thresh2, kernel, iterations=5)

  # Find all contours
  contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key = cv2.contourArea, reverse = True)

  # Find largest contour and surround in min area box
  largestContour = contours[0]
  minAreaRect = cv2.minAreaRect(largestContour)

  # Determine the angle. Convert it to the value that was originally used to obtain skewed image
  angle = minAreaRect[-1]
  if angle < -45:
      angle = 90 + angle
  angle =  -1.0 * angle

  return blur3, angle

def all_process(cropped):
    new_image, angle  = image_cleaning(cropped)
    plt.imsave(cropped_images_path + 'new_image.jpg', new_image)
    im = Image.open(cropped_images_path + 'new_image.jpg')
    borderless_image = trim(im)
    newFilePath = (cropped_images_path + 'borderless.jpg')
    try:
        borderless_image.save(newFilePath)
    except AttributeError:
        print("Couldn't save image {}".format(borderless_image))

    borderless_img = cv2.imread(cropped_images_path + 'borderless.jpg')
    print(int(angle))
    if angle == (-90):
      return borderless_img
    else:
      rotate_image = rotateImage(borderless_img, (-1.0 * angle))
      return rotate_image

###Zeynep###
class PolygonDataset(Dataset):
  def __init__(self):
    print('PolygonDataset start')
    self.counter = 0
        
  def read_image(self, image_name, column_list, type):
    self.image = cv2.imread(train_path + image_name)

    if type == 'box':
      print('BOX')
      self.read_box(self.image, column_list)

    elif type == 'polygon':
      print('POLYGON')
      self.read_polygon(self.image, column_list)

  def read_box(self, image, column_list):
    for i in range(len(column_list)):
      self.counter+=1
      print(column_list[i]['bounding-box'])
      self.title = column_list[i]['title']
      print(self.title)
      x = column_list[i]['bounding-box']['x']
      y = column_list[i]['bounding-box']['y']
      w = column_list[i]['bounding-box']['width']
      h = column_list[i]['bounding-box']['height']
      print(self.image)
      print(int(x), int(y), int(w), int(h))
      return (self.image, int(x), int(y), int(w), int(h))
      #cropped = image[int(self.y):int(self.y)+int(self.h),int(self.x):int(self.x)+int(self.w)]#[y:y+h,x:x+w]

      # result_image = all_process(cropped)
      # plt.imsave(cropped_images_path + f'box_cropped_images/box_{id}_{self.counter}.jpg', result_image)

  def read_polygon(self, image, id, column_list):
    for i in range(len(column_list)):
      self.counter+=1
      self.title = column_list[i]['title']
      print(self.title)
      mask = np.zeros(image.shape[0:2], dtype=np.uint8)
      self.points = column_list[i]['polygon']
      self.points = np.array(self.points,  dtype=np.int32) 

      #method 1 smooth region
      cv2.drawContours(mask, [self.points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        
      #method 2 not so smooth region
      #cv2.fillPoly(mask, points, (255))

      res = cv2.bitwise_and(image,image,mask = mask)
      rect = cv2.boundingRect(self.points) # returns (x,y,w,h) of the rect
      cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

      result_image = all_process(cropped)
      plt.imsave(cropped_images_path + f'polygon_cropped_images/poly_{id}_{self.counter}.jpg', result_image)
