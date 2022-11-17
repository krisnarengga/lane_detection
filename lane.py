import cv2
import numpy as np
import matplotlib.pyplot as plt


from math import atan2, degrees, radians

def get_angle(point_1, point_2): #These can also be four parameters instead of two arrays
    angle = atan2(point_1[1] - point_2[1], point_1[0] - point_2[0])
    #Optional
    angle = degrees(angle)
    # OR
    #angle = radians(angle)
    return angle
    
def make_coordinates(image,line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    
    return np.array([x1,y1,x2,y2])


def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
        
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line, right_line])


def generate_average_lines(image,lines):
    left_fit = []
    right_fit = []
    line_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        line_fit.append((slope,intercept))
        
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
        
    
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line, right_line])
   

def func_canny(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),1)
    canny = cv2.Canny(blur, 50, 100)
    return canny

def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)    
    return line_image

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
    [(100,400),(100,600),(650,600),(650,400)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(img,mask)
    
    #kernel = np.ones((7, 7), np.uint8)
    #img_dilation = cv2.dilate(masked_image, kernel, iterations=1)    
    
    return masked_image
    #return img_dilation

def draw_the_lines(img,lines): 
        
    imge=np.copy(img)     
    
    for line in lines:  
        for x1,y1,x2,y2 in line:
            point_1 = (x1,y1)
            point_2 = (x2,y2)
            
            angle = get_angle(point_1, point_2)
            if (angle > 95 and angle < 150) or (angle > -150 and angle < -95):
                #print(angle)                                    
                cv2.line(imge,(x1,y1),(x2,y2),(0,255,0),thickness=2)
                        
    return imge


cap = cv2.VideoCapture('toll_road.mp4')

lane_changing_flag = 0
counter = 0
mask = 0

while cap.isOpened():
    
    success, frame = cap.read() 
    
    if success:
    
        small_img = cv2.resize(frame,(800,600))
        lane_image = np.copy(small_img)
        canny = func_canny(lane_image)
        
        cropped_image = region_of_interest(canny)
        
        try:
            lines_hough = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength = 60,maxLineGap = 200)
            lane_image = draw_the_lines(lane_image,lines_hough)         
            
            #averaged_lines = generate_average_lines(lane_image,lines_hough)
            #line_image = display_lines(lane_image,averaged_lines)
            #combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
            
        except Exception as e:
            print(e)

        cropped_area = cropped_image[520:540, 320:420]
                
        if (lane_changing_flag == 0):
            cv2.rectangle(lane_image, (320, 520), (420, 540), (0, 255, 0), 0)        
        else:
            cv2.rectangle(lane_image, (320, 520), (420, 540), (0, 0, 255), 0)        
            
        roi_lane = lane_image[521:539, 321:419]
        
        lower_green = np.array([0, 255, 0], dtype = "uint8")
        upper_green = np.array([0, 255, 0], dtype = "uint8")
        
        mask = cv2.inRange(roi_lane,lower_green,upper_green)
        print(mask)
        
        if(255 in mask):
            lane_changing_flag = 1
        
        if (lane_changing_flag == 1):
            cv2.putText(lane_image,'PERHATIAN ANDA SEDANG BERPINDAH JALUR',(100,200), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(lane_image,'(ATTENTION YOU ARE CHANGING LANE !!!)',(120,250), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)            
        else:
            cv2.putText(lane_image,'ANDA BERADA DI DALAM JALUR',(200,200), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),3,cv2.LINE_AA)
            cv2.putText(lane_image,'(YOU ARE ON THE TRACK)',(230,250), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),3,cv2.LINE_AA)
                    
        
        if (counter % 200 == 0):
            lane_changing_flag = 0
            counter = 0
        
        counter = counter + 1
        
        
        cv2.imshow("canny edge",canny)
        cv2.imshow("original",lane_image)
        cv2.waitKey(10)
        
    else :
        break
        
cap.release()
cv2.destroyAllWindows() 
    