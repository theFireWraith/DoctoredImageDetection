import PIL
from PIL import Image
from matplotlib import pyplot as plt

i=input("enter the number of grayscale images")
i=int(i)
#i=i-1
l=0
a=0
k=0
filelist1=[]
filelist2=[]
#print(i)
for j in range(i):
        #j=j+1
        filelist1.append('gray'+str(j)+'.jpg')
        filelist2.append('grayori'+str(j)+'.jpg')
        #print(filelist1)
for imagefile1 in filelist1:
          im3=Image.open(imagefile1)
          wi, he = im3.size
          #print(im3.size)
          colors = im3.getcolors(wi*he) #Returns a list [(pixel_count, (R, G, B))]
          #print(colors)
          a=a+1
          def hexencode(rgb):
              r=rgb[0]
              g=rgb[1]
              b=rgb[2]
              return '#%02x%02x%02x' %(r,g,b)
          for idx, c in enumerate(colors):
              plt.bar(idx, c[0], color=hexencode(c[1]))
          out2='hist'+str(l)+'.jpg'
          l=l+1
          plt.savefig(out2)
          wi=he=0
        
for imagefile2 in filelist2:
          im4=Image.open(imagefile2)
          wi, he = im4.size
          #print(im3.size)
          colors = im4.getcolors(wi*he) #Returns a list [(pixel_count, (R, G, B))]
          #print(colors)
          a=a+1
          def hexencode(rgb):
              r=rgb[0]
              g=rgb[1]
              b=rgb[2]
              return '#%02x%02x%02x' %(r,g,b)
          for idx, c in enumerate(colors):
              plt.bar(idx, c[0], color=hexencode(c[1]))
          out3='orihist'+str(k)+'.jpg'
          k=k+1
          plt.savefig(out3)
