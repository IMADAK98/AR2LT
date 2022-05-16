import cv2
import matplotlib.pyplot as plt

image = "images/download.jpg"
img = cv2.imread(image)


# display full size image function
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width = im_data.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()


# displaying the orginal function
display(image)

# inverting image

inverted_image = cv2.bitwise_not(img)
cv2.imwrite("temp/Inverted.jpg", inverted_image)
display("temp/Inverted.jpg")  # displaying the inverted function


# Binarizaion
#1-gray conversion

def grayscale(image):
 return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


gray_image = grayscale(img)
cv2.imwrite("temp/Gray.jpg", gray_image)
display("temp/Gray.jpg")

#2-binary conversion

thresh,im_bw=cv2.threshold(gray_image,120,255,cv2.THRESH_BINARY)
cv2.imwrite("temp/bw_image.jpg",im_bw)
display("temp/bw_image.jpg")

#3-Noise removal

def noise_removal(image):
    import numpy as np
    kernel=np.ones((1,1),np.uint8)
    image=cv2.dilate(image,kernel,iterations=1)
    kernel=np.ones((1,1),np.uint8)
    image=cv2.erode(image,kernel,iterations=1)
    image=cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
    image=cv2.medianBlur(image,3)
    return (image)
no_noise=noise_removal(im_bw)
cv2.imwrite("temp/no_noise.jpg",no_noise)
display("temp/no_noise.jpg")