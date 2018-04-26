# Ad-hoc algorithm for copy-move forgery detection in images.
# Implemented by - vasiliauskas.agnius@gmail.com
# Robust match algorithm steps:
#  1. Blur image for eliminating image details
#  2. Convert image to degraded palette
#  3. Decompose image into small NxN pixel blocks
#  4. Alphabetically order these blocks by their pixel values
#  5. Extract only these adjacent blocks which have small absolute color difference
#  6. Cluster these blocks into clusters by intersection area among blocks
#  7. Extract only these clusters which are bigger than block size
#  8. Extract only these clusters which have similar cluster, by using some sort of similarity function (in this case Hausdorff distance between clusters)
#  9. Draw discovered similar clusters on image
import PIL
import sys
from PIL import Image, ImageFilter, ImageDraw
import operator as op
from optparse import OptionParser
from matplotlib import pyplot as plt

imag=Image.open('original.jpg')
def Dist(p1,p2):
 """
 Euclidean distance between 2 points
 """
 x1, y1 = p1
 x2, y2 = p2
 return (((x1-x2)*(x1-x2)) + ((y1-y2)*(y1-y2)))**0.5

def intersectarea(p1,p2,size):
 """
 Given 2 boxes, this function returns intersection area
 """
 x1, y1 = p1
 x2, y2 = p2
 ix1, iy1 = max(x1,x2), max(y1,y2)
 ix2, iy2 = min(x1+size,x2+size), min(y1+size,y2+size)
 iarea = abs(ix2-ix1)*abs(iy2-iy1)
 if iy2 < iy1 or ix2 < ix1: iarea = 0
 return iarea

def Hausdorff_distance(clust1, clust2, forward, dir):
 """
 Function measures distance between 2 sets. (Some kind of non-similarity between 2 sets if you like).
 It is modified Hausdorff distance, because instead of max distance - average distance is taken.
 This is done for function being more error-prone to cluster coordinates.
 """
 if forward == None:
  return max(Hausdorff_distance(clust1,clust2,True,dir),Hausdorff_distance(clust1,clust2,False,dir))
 else:
  clstart, clend = (clust1,clust2) if forward else (clust2,clust1)
  dx, dy = dir if forward else (-dir[0],-dir[1])
  return sum([min([Dist((p1[0]+dx,p1[1]+dy),p2) for p2 in clend]) for p1 in clstart])/len(clstart)

def hassimilarcluster(ind, clusters):
 """
 For given cluster tells does it have twin cluster in image or not.
 """
 item = op.itemgetter
 global opt
 found = False
 tx = min(clusters[ind],key=item(0))[0]
 ty = min(clusters[ind],key=item(1))[1]
 for i, cl in enumerate(clusters):
  if i != ind:
   cx = min(cl,key=item(0))[0]
   cy = min(cl,key=item(1))[1]
   dx, dy = cx - tx, cy - ty
   specdist = Hausdorff_distance(clusters[ind],cl,None,(dx,dy))
   if specdist <= int(opt.rgsim):
    found = True
    break
 return found

def blockpoints(pix, coords, size):
 """
 Generator of pixel colors of given block.
 """
 xs, ys = coords
 for x in range(xs,xs+size):
  for y in range(ys,ys+size):
   yield pix[x,y]

def colortopalette(color, palette):
 """
 Convert given color into palette color.
 """
 for a,b in palette:
  #print(a,b,color)
  if color >= a and color < b:
   #print(b)
   return b

def imagetopalette(image, palcolors):
 """
 Convert given image into custom palette colors
 """
 assert image.mode == 'L', "Only grayscale images supported !"
 pal = [(palcolors[i],palcolors[i+1]) for i in range(len(palcolors)-1)]
 image.putdata([colortopalette(c,pal) for c in list(image.getdata())])

def getparts(image, block_len):
 """
 Decompose given image into small blocks of data.
 """
 img = image.convert('L') if image.mode != 'L' else image
 w, h = img.size 
 parts = []
 # Bluring image for abandoning image details and noise.
 global opt
 for n in range(int(opt.imblev)):
  img = img.filter(ImageFilter.SMOOTH_MORE)
 # Converting image to custom palette
 imagetopalette(img, [x for x in range(256) if x%int(opt.impalred) == 0])
 pix = img.load()
 
 for x in range(w-block_len):
  for y in range(h-block_len):
   data = list(blockpoints(pix, (x,y), block_len)) + [(x,y)]
   parts.append(data)
 parts = sorted(parts)
 return parts

def similarparts(imagparts):
 """
 Return only these blocks which are similar by content.
 """
 dupl = []
 global opt
 l = len(imagparts[0])-1
 
 for i in range(len(imagparts)-1):  
  difs = sum(abs(x-y) for x,y in zip(imagparts[i][:l],imagparts[i+1][:l]))
  #print(difs)
  mean = float(sum(imagparts[i][:l])) / l
  dev = float(sum(abs(mean-val) for val in imagparts[i][:l])) / l
  if dev/mean >= float(opt.blcoldev):
   if difs <= int(opt.blsim):
    if imagparts[i] not in dupl:
     dupl.append(imagparts[i])
    if imagparts[i+1] not in dupl:
     dupl.append(imagparts[i+1])
 return dupl

def clusterparts(parts, block_len):
 """
 Further filtering out non essential blocks.
 This is done by clustering blocks at first and after that
 filtering out small clusters and clusters which doesn`t have
 twin cluster in image.
 """
 parts = sorted(parts, key=op.itemgetter(-1))
 global opt
 clusters = [[parts[0][-1]]]
 print(clusters)
 
 # assign all parts to clusters
 for i in range(1,len(parts)):
  x, y = parts[i][-1]
  
  # detect box already in cluster
  fc = []
  for k,cl in enumerate(clusters):
   for xc,yc in cl:
    ar = intersectarea((xc,yc),(x,y),block_len)
    intrat = float(ar)/(block_len*block_len)
    if intrat > float(opt.blint):
     if not fc: clusters[k].append((x,y))
     fc.append(k)
     break
  
  # if this is new cluster
  if not fc:
   clusters.append([(x,y)])
  else:
   # re-clustering boxes if in several clusters at once
   while len(fc) > 1:
    clusters[fc[0]] += clusters[fc[-1]]
    del clusters[fc[-1]]
    del fc[-1]
 
 item = op.itemgetter
 # filter out small clusters
 clusters = [clust for clust in clusters if Dist((min(clust,key=item(0))[0],min(clust,key=item(1))[1]), (max(clust,key=item(0))[0],max(clust,key=item(1))[1]))/(block_len*1.4) >= float(opt.rgsize)]
 
 # filter out clusters, which doesn`t have identical twin cluster
 clusters = [clust for x,clust in enumerate(clusters) if hassimilarcluster(x,clusters)]
 
 return clusters

def marksimilar(image, clust, size):
 """
 Draw discovered similar image regions.
 """
 im7=Image.open(args[0])
 i=0
 global opt
 blocks = []
 if clust:
  draw = ImageDraw.Draw(image)
  mask = Image.new('RGB',(size,size))
  for cl in clust:
   for x,y in cl:
   	im = image.crop((x,y,x+size,y+size))
   	im = Image.blend(im,mask,0.5)
   	blocks.append((x,y,im))
  for bl in blocks:
  	x,y,im = bl
  	image.paste(im,(x,y,x+size,y+size))
  if int(opt.imauto):
   for cl in clust:
      cx1 = min([cx for cx,cy in cl])
      cy1 = min([cy for cx,cy in cl])
      cx2 = max([cx for cx,cy in cl]) + block_len
      cy2 = max([cy for cx,cy in cl]) + block_len
      box=(cx1, cy1, cx2, cy2)
      area=im7.crop(box)
      area1=imag.crop(box)
      out1='ori'+str(i)+'.jpg'
      area.convert('LA')
      out = 'crop'+str(i)+'.jpg'
      i=i+1
      area.save(out)
      area1.save(out1)
      draw.rectangle([cx1,cy1,cx2,cy2],outline="magenta")
      k=0
      filelist=[]
      filelist1=[]
      filelist2=[]
      print(i)
      for j in range(i):
        #j=j+1
        filelist.append('crop'+str(j)+'.jpg')
      print(filelist)
      j=0
      l=0
      colours=0
      print(filelist[0])
      '''for imagefile in filelist:
       # im2=Image.open(imagefile).convert('L')
        #out1='gray'+str(k)+'.jpg'
        #k=k+1
        #im2.save(out1)'''
      '''for j in range(i):
        j=j+1
        filelist1.append('gray'+str(j-1)+'.jpg')
        filelist2.append('ori'+str(j-1)+'.jpg')
        print(filelist1)
        print(filelist2)
      for imagefile1 in filelist1:
          im3=Image.open(imagefile1)
          wi, he = im3.size
          print(im3.size)
          colors = im3.getcolors(wi*he) #Returns a list [(pixel_count, (R, G, B))]
          print(colours)
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
          v=0

          for imagefile2 in filelist2:
                im3=Image.open(imagefile2)
                wi, he = im3.size
                print(im3.size)
                colors = im3.getcolors(wi*he) #Returns a list [(pixel_count, (R, G, B))]
                #print(colours)
          def hexencode(rgb):
              r=rgb[0]
              g=rgb[1]
              b=rgb[2]
              return '#%02x%02x%02x' %(r,g,b)
          for idx, c in enumerate(colors):
              plt.bar(idx, c[0], color=hexencode(c[1]))
          out3='orihist'+str(v)+'.jpg'
          v=v+1
          plt.savefig(out3)'''

 return image


     
    

if __name__ == '__main__':
 cmd = OptionParser("usage: %prog image_file [options]")
 cmd.add_option('', '--imauto', help='Automatically search identical regions. (default: %default)', default=1)
 cmd.add_option('', '--imblev',help='Blur level for degrading image details. (default: %default)', default=8)
 cmd.add_option('', '--impalred',help='Image palette reduction factor. (default: %default)', default=15)
 cmd.add_option('', '--rgsim', help='Region similarity threshold. (default: %default)', default=5)
 cmd.add_option('', '--rgsize',help='Region size threshold. (default: %default)', default=1.5)
 cmd.add_option('', '--blsim', help='Block similarity threshold. (default: %default)',default=200)
 cmd.add_option('', '--blcoldev', help='Block color deviation threshold. (default: %default)', default=0.2)
 cmd.add_option('', '--blint', help='Block intersection threshold. (default: %default)', default=0.2)
 opt, args = cmd.parse_args()
 if not args:
  cmd.print_help()
  sys.exit()
 print ('Analyzing image, please wait... (can take some minutes)')
 block_len = 15
 im = Image.open(args[0])
 lparts = getparts(im, block_len)
 dparts = similarparts(lparts)
 cparts = clusterparts(dparts, block_len) if int(opt.imauto) else [[elem[-1] for elem in dparts]]
 im = marksimilar(im, cparts, block_len)
 out = args[0].split('.')[0] + '_analyzed.jpg'
 im.save(out)
 #hist(i)
 print ('Done. Found', len(cparts) if int(opt.imauto) else 0, 'identical regions')
 print ('Output is saved in file -', out)
