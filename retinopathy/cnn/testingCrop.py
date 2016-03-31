import numpy as np
import cv2
import csv

def crop_img(image, ratio=.75):
    
    passed = False
    hadTrouble = False
    
    image = cv2.bilateralFilter(image, 9, 75, 75)
    
    output = image.copy()
    
    #Some of the images are darker than others so it is necessary to loop through
    #various threshold values until a proper bounding rectangle is found
    for i in range(0,10):
        
        ret,thresh = cv2.threshold(image,5+(5*i),255,cv2.THRESH_BINARY)
        thresh1 = thresh.copy()
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        brx,bry,brw,brh = cv2.boundingRect(cnt)
        if(brw > 100 and brh > 100):
            print image_name + " " + str(brw) + " " + str(brh) + " " + str(5+(5*i))
            passed = True
            if hadTrouble:
                cv2.rectangle(output,(brx,bry),(brx+brw,bry+brh),(0,255,0),2)
                cv2.imshow("debug", output)
                cv2.waitKey(0)
            break
        else:
            hadTrouble = True
            cv2.imshow("debug", thresh1)
            cv2.waitKey(0)
            
    if not passed:
        print "FAILED"
        return (image, False)
    
        
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    
    cv2.circle(output,center,radius,(0,0,255),2)
    cv2.circle(output,center,int(radius*2/3),(0,0,255),1)
    cv2.circle(output,center,int(radius/3),(0,0,255),1)
    cv2.circle(output,center,2,(0,0,255),2)

    height, width = image.shape
    newY = max(int(y-(radius*ratio)), 0)
    if newY == 0:
        print "caught top"
    newHeight = int(radius*ratio*2)
    if newY + newHeight > height:
        newHeight = height - newY
        print "caught bot"
    cv2.rectangle(output,(brx,newY),(brx+brw,newY+newHeight),(255,0,0),2)
    
    #NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    crop_img = image[newY:newY+newHeight, brx:brx+brw]
    #crop_img = cv2.resize(crop_img, (resizeWidth, resizeHeight))
    cv2.imshow("debug", output)
    cv2.waitKey(0)
    return (crop_img, True)
    
    
    

if __name__ == "__main__":    
    data_images = []
    labels = []
    size = 500
    div = 3
    image_name = "/root/project/train/766_left.jpeg"
    image = cv2.imread(image_name)
    height, width, channels = image.shape
    #image = cv2.resize(image, (int(round(width/div)), int(round(height/div))))
    #output = image.copy()
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #img, bool = crop_img(gray)
    img = image.copy()
    img = cv2.resize(img, (700, 500))
    with open(image_name, 'rb') as f:
        data = f.read()
    with open('766_left.jpeg', 'wb') as f:
        f.write(data)
    #cv2.imshow(image_name, img)
    #cv2.waitKey(0)
    
    
    
    '''passed = False
    hadTrouble = False
    
    for i in range(0,15):
        ret,thresh = cv2.threshold(gray,5+(5*i),255,cv2.THRESH_BINARY)
        thresh2 = thresh.copy()
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        if(w > 100 and h > 100):
            #print image_name + " " + str(w) + " " + str(h) + " " + str(5+(5*i))
            passed = True
            if hadTrouble:
                cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.imshow(image_name, output)
                cv2.waitKey(0)
            break
        else:
            hadTrouble = True
            cv2.imshow(image_name, thresh2)
            cv2.waitKey(0)
            
    if not passed:
        print "didn't pass"
    
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    
    cv2.circle(output,center,radius,(0,0,255),2)
    
    ellipse = cv2.fitEllipse(cnt)
    
    cv2.ellipse(output,ellipse,(255,0,0),2)
    
    
    cv2.imshow(image_name, output)
    #cv2.imshow("image", np.hstack([thresh, output]))
    cv2.waitKey(0)'''
                 
                
