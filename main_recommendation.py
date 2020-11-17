from conf.yolo import YOLO
from conf.recommendation import recommendation
import keras.backend as K

K.clear_session()
yolo = YOLO()

#To test for an image
import cv2
# image = 'test_files/test2.jpg'
image = cv2.imread('test_files/test2.jpg', cv2.IMREAD_COLOR)
r_image, ObjectsList = yolo.detect_img(image)
print(ObjectsList)
yolo.close_session()

cv2.imwrite("output/output.jpg", r_image)

# To test for video
video = 'test_files/test.jpg'
yolo.detect_video(video)

'''
Until here, we have detected the item in the video (football and helmet). A pop up will be generated in the detected
footballs prompting the user to buy the item. If use clicks/buy the football then we can recommend other content (items)
similar to football.

-> For instance, We assume that user has clicked/bought from football popup.
'''
user_click = "football"
recommended_items = recommendation(user_click)
for i in range(3):
    print(f"Rank {i}: {recommended_items[i]}")
