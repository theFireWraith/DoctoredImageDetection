import PIL
from PIL import Image
from matplotlib import pyplot as plt

im = Image.open('grayori1.jpg') 
w, h = im.size
#print(im.size)
colors = im.getcolors(w*h) #Returns a list [(pixel_count, (R, G, B))]
def hexencode(rgb):
    #print(rgb)
    r=rgb[0]
    g=rgb[1]
    b=rgb[2]
    return '#%02x%02x%02x' % (r,g,b)

for idx, c in enumerate(colors):
    plt.bar(idx, c[0], color=hexencode(c[1]))

plt.savefig('grayoriout1.jpg')
