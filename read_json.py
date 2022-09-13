import numpy as np
import json
from pprint import pprint
import cv2

from shapely.geometry import Point, Polygon

from read_image import PolygonDataset

#you should give your local path
train_path = 'D:/Cyberneticlabs/CoWork/train/'
annotation_path = 'D:/Cyberneticlabs/CoWork/news_paper_segmentation_-export-2022-09-06T10_59_00.764Z.json'

class rearrenge_column :  

  def __init__(self):
    self.blockes_polygon = []
    self.blockes_box = []

  def rearrange_column_box(self,box_column_list: list):

    for title_column in box_column_list : 
      if title_column["title"]=="Content Title " :
        title = title_column
        self.blockes_box.append(title)
        del  box_column_list[(box_column_list.index(title))]   
   
    for i in range(len(box_column_list)):
      max_x = box_column_list[0]["bounding-box"]["x"]
      max_y = box_column_list[0]["bounding-box"]["y"]
      max_width =  box_column_list[0]["bounding-box"]["width"]
      block = box_column_list[0]
      
      for column in box_column_list :      
        if (column["bounding-box"]["x"] >= max_x  ):
          block = column
          max_x = column["bounding-box"]["x"]
          max_y = column["bounding-box"]["y"]
        elif (column["bounding-box"]["x"] < max_x and column["bounding-box"]["y"] > max_y and column["bounding-box"]["width"] >= max_width) :
          block = column 
          max_width = column["bounding-box"]["width"]
          max_y = column["bounding-box"]["y"]
      self.blockes_box.append(block)
      del box_column_list[(box_column_list.index(block))]
    
    #print('rearrange_column_box')
    #pprint(self.blockes_box)
    return(self.blockes_box)

  def rearrange_column_polygon(self,polygon_column_list: list): 

    for title_column in polygon_column_list : 
      if title_column["title"]== "Content Title polygon":
        title = title_column
        self.blockes_polygon.append(title)
        del  polygon_column_list[(polygon_column_list.index(title))]   
   

    for i in range(len(polygon_column_list)): 

      max_x = rearrenge_column.polygon_max_x_(polygon_column_list[0])
      max_y = rearrenge_column.polygon_max_y_(polygon_column_list[0])
      block = polygon_column_list[0]

      for column in polygon_column_list :
        if (rearrenge_column.polygon_max_x_(column)> max_x  and rearrenge_column.polygon_max_y_(column)<= max_y ):
          block = column
          max_x = rearrenge_column.polygon_max_x_(column)
          max_y = rearrenge_column.polygon_max_y_(column)
         
        elif (rearrenge_column.polygon_max_x_(column)< max_x and rearrenge_column.polygon_max_y_(column) < max_y) : 
          block = column 
      

      self.blockes_polygon.append(block)
      del polygon_column_list[(polygon_column_list.index(block))]
      
    # print('rearrange_column_polygon')
    # pprint(self.blockes_polygon)
    return(self.blockes_polygon)

  def polygon_max_x_(box : dict):
    list = []
    for coordinate in box["polygon"] :
      list.append(coordinate[0])
    max_x = np.max(list)
    return max_x
    

  def polygon_max_y_(box : dict):
    list = []
    for coordinate in box["polygon"] :
      list.append(coordinate[1])
    max_y = np.max(list)
    return max_y

"""mark columns  for each polygon and box """

class mark_column_type: 

  def __init__(self):
    self.polygon_content = []
    self.box_content = []
   
  def mark_column_polygon(self,content_polygon_block , image_box):
    
    for box in image_box['tasks'][0]['objects']:
      if box["title"] == "Content Title polygon" or box["title"] == "Column polygon" or box["title"] =="author polygon" or box["title"] == "image polygon" :
        self.polygon_content.append(box) if Polygon(content_polygon_block["polygon"]).contains(Polygon(box['polygon'])) == True else None
    # print('mark_column_polygon')
    # pprint(self.polygon_content)
    return rearrenge_column().rearrange_column_polygon(self.polygon_content)
          
  def mark_column_box(self,content_box_block :list , image_box):

    box_x = content_box_block['bounding-box']['x']
    box_y = content_box_block['bounding-box']['y']
    box_height = content_box_block['bounding-box']['height']
    box_width = content_box_block['bounding-box']['width']
    
    for box in image_box['tasks'][0]['objects']:
      if box["title"] == 'Content Title '  or box['title'] == 'Column'  or box["title"] == "Image"  or box["title"] =="author" : 
        # If top-left inner box corner is inside the bounding box  
          if box_x < box['bounding-box']['x'] and box_y < box['bounding-box']['y']:
          # If bottom-right inner box corner is inside the bounding box
            if box['bounding-box']['x'] + box['bounding-box']['width'] < box_x + box_width and box['bounding-box']['y'] + box['bounding-box']['height'] < box_y + box_height:
              self.box_content.append(box)

    if (len(self.box_content) == 0  )  :
      # print('asdfghjklşi')
      # pprint([content_box_block])
      return [content_box_block]

    else :  
      # print('box_content')
      # pprint(self.box_content)
      return rearrenge_column().rearrange_column_box(self.box_content)
  
  def mark_ads(self,ads_block):
    # print([ads_block])
    return([ads_block])

  def mark_title(self,title_block):
    # print([title_block])
    return([title_block])



def read_box(image_name, column_list):
  points = []

  for i in range(len(column_list)):
    image = cv2.imread(str(train_path) + str(image_name))
    title = column_list[i]['title']
    x = column_list[i]['bounding-box']['x']
    points.append(int(x))
    y = column_list[i]['bounding-box']['y']
    points.append(int(y))
    w = column_list[i]['bounding-box']['width']
    points.append(int(w))
    h = column_list[i]['bounding-box']['height']
    points.append(int(h))
    #points.append((int(x), int(y), int(w), int(h)))
  
  # print(points)
  # print(type(points))
  # print(points[0])
  # print(type(points[0]))
  return image, points


def read_polygon(image_name, column_list):
  points = []
  image = cv2.imread(str(train_path) + str(image_name))
  for i in range(len(column_list)):
    title = column_list[i]['title']
    point = column_list[i]['polygon']
    point = np.array(point,  dtype=np.int32)
    points.append(point)
  
  # print(points)

  return image, points

class Test():

  # def __init__(self):
  #   #self.read_image = PolygonDataset()
  #   self.main()

  def main ():
    images = []
    bboxes = []
    imagePaths = []
    f = open(annotation_path)
    data = json.load(f)
    for i in range(len(data)):
      image_box = data[i]
      image_name = image_box["externalId"]
      #image_id = '96-04-11-0014_png.rf.fba46cbf74bedce40200217092324542.jpg'
      #print('IMGAE: ' + str(i) + ' / ' + 'IMAGE NAME: ' + image_name)
      imagePaths.append(str(train_path) + str(image_name))
      contents = []
      counter = 0

      for box in image_box['tasks'][0]['objects']:
        if box["title"] == "Content" or  box["title"] == "Content polygon":
          contents.append(box)
        
        elif box["title"] == "advertisement"  :
          advertisement_list = mark_column_type().mark_ads(box)  
          if "polygon" in advertisement_list[0].keys():
            image, points = read_polygon(image_name, advertisement_list)
          else:
            image, points = read_box(image_name, advertisement_list)
            # print(image_name)
            # print(points)
            
          images.append(image)
          bboxes.append(points) 
        '''
        elif box["title"] == "title":
          title_list = mark_column_type().mark_title(box)
          if "polygon" in title_list[0].keys():
            image, points = read_polygon(image_name, title_list)
          else:
            image, points = read_box(image_name, title_list) 
          images.append(image)
          bboxes.append(points)
      
      for content in contents :  
        counter+=1
        print("content" + str(counter))
        
        if "polygon" in content.keys():
          pprint(content)
          polygon_column_list = mark_column_type().mark_column_polygon(content,image_box)
          # print('Polygon List')
          # pprint(polygon_column_list)
          image, points = read_polygon(image_name, polygon_column_list)
          images.append(image)
          bboxes.append(points)
        
        if 'bounding-box' in content.keys():
          box_column_list = mark_column_type().mark_column_box(content,image_box) 
          # print('Box List')
          # pprint(box_column_list)
          if(box_column_list is None):
            print("List is empty")    
          else:
            image, points = read_box(image_name, box_column_list)
            images.append(image)
            bboxes.append(points)
        '''
        
    # print(images)
    # print(bboxes)
    return images, bboxes

# my_test = Test()
# print('App is initializing...')




    
    