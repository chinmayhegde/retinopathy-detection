import numpy as np
import cv2
import csv
#from datetime import datetime

#params
#   image: grayscale image before resize
#   ratio: decimal value giving the portion of the vertical radius to keep
#   resizeHeight/resizeWidth: int values to resize the image to after
#return 
#   (cropped image, success boolean)
def crop_img(image, ratio=.75, resizeHeight= 518, resizeWidth=718):
    
    #reduce size proportionally
    div=3
    height, width = image.shape
    image = cv2.resize(image, (int(round(width/div)), int(round(height/div))))
    img = image.copy()
    
    passed = False
    
    #Some of the images are darker than others so it is necessary to loop through
    #various threshold values until a proper bounding rectangle is found
    for i in range(0,15):
        ret,thresh = cv2.threshold(img,10+(5*i),255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        brx,bry,brw,brh = cv2.boundingRect(cnt)
        if(brw > 100 and brh > 100):
            passed = True
            break
        else:
            #blur if needed (save time not blurring every image, only a few need it)
            img = cv2.bilateralFilter(img, 9, 75, 75)
            
    if not passed:
        print "crop failed"
        return (image, False)
    
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)

    height, width = image.shape
    newY = max(int(y-(radius*ratio)), 0)
    newHeight = int(radius*ratio*2)
    if newY + newHeight > height:
        newHeight = height - newY
    
    #NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    crop_img = image[newY:newY+newHeight, brx:brx+brw]
    crop_img = cv2.resize(crop_img, (resizeWidth, resizeHeight))
    return (crop_img, True)
        

if __name__ == "__main__":
    # Get image names and classifications
    names = {i: [] for i in range(5)}
    with open('subsetcsv.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            names[int(row[1])].append('train\\' + row[0] + '.jpeg')

    #print "\nstart " + str(datetime.now())
    for classification, image_names in names.iteritems():
        for image_name in image_names:
            # median image size in (2592, 3888)
            image = cv2.imread(image_name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img, bool = crop_img(gray)
            if(not bool):
                print image_name
       
            cv2.imshow(image_name, img)
            cv2.waitKey(0)
            
    #print "end " + str(datetime.now())
                 
                
