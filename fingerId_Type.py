#!/usr/bin/python

import cv2, os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
import math
from sklearn.preprocessing import normalize

import collections
import json


_waitingtime = 0.5
SIZE = 300
a = 150.0
Y = 95.0
kernel = np.ones((5,5),np.float32)/25
#kernel = np.ones((5,5),np.float32)
G = 100


def imprime_blocos(xy, H, block, w, inter):

	w2 = int(w / 2)
	min_radius = (-w2)+1
	max_radius = w2-1

	for i in range(0, H):

		for j in range(0, H):
						
			if ( inter[i,j] == 0 ):
				ang  = xy[i,j]
				coss = math.cos(ang)
				sen  = math.sin(ang)
		
				ni = i*w+w2
				nj = j*w+w2

				#min_radio = min_radius + (int(save[i,j]) / 2)
				#max_radio = max_radius - (int(save[i,j]) / 2)
				#min_radio = min_radius * (int(save[i,j]) / 2)
				#max_radio = max_radius * (int(save[i,j]) / 2)

				#print i, j, min_radio, max_radio, save[i,j]
							
				for r in range(min_radius, max_radius):
					xt = int(nj + r*coss)
					yt = int(ni + r*sen)

					block[yt,xt] = 0
					#print i, j, xt, yt



def delta_zuero(a,b):
	diff = a - b
	
	if ( abs(diff) < math.pi/2 ):
		return diff
	if ( diff <= ((-1)*math.pi)/2 ):
		return math.pi + diff
	else: 
		return math.pi - diff



# ************************* Main *************************** #
if ( len(sys.argv) < 2 ):
	print "python fingerId_Type.py imagesPath [resutlPath]"
	exit(-1)

path = sys.argv[1]

if ( len(sys.argv) > 2 ):
    comp_path = sys.argv[2]

images = np.array([])
label = np.array([])
all_core = np.array([])
ang_core = np.array([])
all_delta = np.array([])


subject_path = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.raw') and os.path.isfile(os.path.join(path,f)) ]
subject_path.sort()
#print subject_path

for image_path in subject_path:

		print ''
		print '***********************************************************'
		print 'Agora: ', image_path

		image = np.fromfile(image_path, dtype='uint8', sep="")
		image = image.reshape([SIZE, SIZE])
		A = image
		
		label_now =  image_path.split('/')[1]
		label_now = (label_now.split('R')[0]).split('f')[1]
		label = np.append( label, label_now )


		# Support that A (i, j) is image gray level at pixel (i, j), 
		#   u and s2(quadrado) are the mean and variance of gray levels of input image, 
		#   and a=150, y=95, y must satisfy y>s.
		#
		# B(i,j) <- a + y * ( [A(i,j) - u] / s ) 

		u = np.mean(image)
		s = np.std(image)
		print u, s
	   
		B = np.zeros([SIZE, SIZE])
		for i in range( SIZE ):
			for j in range( SIZE ):
				B[i,j] = a + Y * (( float(A[i,j]) - u) / s) 
				 
		
		# ************************** Orientation Computation ******************************* #
		#img = cv2.medianBlur(B, 5)
		blur = cv2.filter2D(B,-1,kernel)
		img = np.copy(blur)

		# sobel and a = 300x300  #3
		sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
		sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
	   

		# Double the angles of the gradient vectors before averaging each block...
		ax = np.subtract(np.square(sobelx), np.square(sobely))
		ay = np.multiply((2 * sobelx), sobely)
		print 'ax', ax.shape
		
	   
		# Average gradient em cada bloco R[wxw] 
		w = 10
		w2d = 1.0 / float(w * w)
		amx = np.array([])
		amy = np.array([])
		for i in range(0, SIZE, w):
			tmp_x = np.array([])
			tmp_y = np.array([])
			for j in range(0, SIZE, w):
				sx = 0.0
				sy = 0.0
				
				for k in range ( i, i+w ):
					for l in range ( j, j+w ):
						sx += ax[k,l]
						sy += ay[k,l]
						#print k, l, sx, sy, ax[k,l], ay[k,l]

				tmp_x = np.append( tmp_x, (sx * w2d) )
				tmp_y = np.append( tmp_y, (sy * w2d) )
			
			amx = np.concatenate( (amx, tmp_x) ) 
			amy = np.concatenate( (amy, tmp_y) )

		Sw = SIZE / w

		amx = np.reshape(amx, (Sw,Sw))
		amy = np.reshape(amy, (Sw,Sw))
		print 'amx = ', amx.shape
		
		#print amx

		# Definindo angulo por bloco
		
		
		
		# Imprimir baseado no modulo do angulo
		#   hue = sqrt((y*y)+(x*x))
		# radius * hue
		#   -> imprime um ponto se for sem angulo, angulo pequeno
		
		
		xy = np.array([])
		save = np.array([])
		min_size = len( amx )
		for i in range( min_size ):
			for j in range( min_size ):
				x = amx[i,j]
				y = amy[i,j]
				xy = np.append( xy , math.atan2(y, x) )
				save = np.append( save, math.sqrt((y*y) + (x*x)) )
				#save = np.append( save, math.sqrt(math.atan2(y, x) * math.atan2(y, x)) )

		xy = np.reshape(xy, (Sw,Sw))
		print 'xy = ', xy.shape
		# amx, amy, xy = 30x30
		save = np.reshape(save, (Sw,Sw))
		
		
		xy = (xy / 2) + (math.pi / 2)
		block = np.copy(img)

		H = (SIZE / w)
		
		#imprime_blocos(xy, H, block, w)


		# *********************** Region of Interest Detection **************************** #
		
		# v = w0 (1 - u) + w1 * sigma + w2
		#   w0 = 0.5 | w1 = 0.5 | w2 is the ratio of distance to the center of the fingerprint image | u and sigma are normalized to be in [0,1] | if v > 0.8 the block is what we want

		#norm = normalize(img)
		#u = np.mean(norm) 
		#s = np.std(norm)
		c = (H / 2) 
		w0 = 0.5
		w1 = 0.5


		# Media de todos os blocos; normaliza pelo maior valor. 
		#   Depois volta calculando  ( em vez de /255 )
	   

		all_mean = np.zeros([H,H])
		all_std = np.zeros([H,H])
		for i in range( H ):
			for j in range( H ):
				ni = i*w
				ei = i*w+w 
				nj = j*w
				ej = j*w+w

				bloco = img[ni:ei, nj:ej]
				all_mean[i,j] = np.mean(bloco)
				all_std[i,j] = np.std(bloco)
				
		max_mean = np.amax(all_mean)
		max_std = np.amax(all_std)
		#print max_mean, max_std


		inter = np.zeros([H,H])        
		for i in range( H ):
			for j in range( H ):
				di = float(i) - float(c)
				dj = float(j) - float(c)
				w2 = ( math.sqrt((float(di*di + dj*dj))) / float(H) ) 

				ni = i*w
				ei = i*w+w 
				nj = j*w
				ej = j*w+w

				bloco = img[ni:ei, nj:ej]
				u = np.mean(bloco)/max_mean
				s = np.std(bloco)/max_std
			
				v = w0 * (1 - u) + w1 * s + 1-w2
					
				if ( v < 0.8 ):
				    for k in range ( ni, ni+w ):
					for l in range ( nj, nj+w ):
					    img[k,l] = G 
				    inter[i,j] = G
	   
	  
		imprime_blocos(xy, H, block, w, inter)


		# ***************************** Singular Point Detection *********************************** #

		#ota_img = np.copy(block)
		#ota_img = np.copy(img)

		# Poincare'

		p = np.zeros([H, H])
		for i in range( 1, H-1 ):
			for j in range( 1, H-1 ):

				bloco_x = amx[(i-1):(i+1), (j-1):(j+1)]
				bloco_y = amy[(i-1):(i+1), (j-1):(j+1)]

				b_a = np.sum(bloco_x) + amx[i,j]
				b_b = np.sum(bloco_y) + amy[i,j]


				p[i,j] = (math.atan2(b_b, b_a) + np.pi) / 2.0

				#p[i,j] = (math.atan2(b_b, b_a))
				#p[i,j] = (p[i,j] / 2.0) + (math.pi / 2.0)
				#print i,j, p[i,j]
				

		ota_img = np.copy(img)

		print p.shape
		#print p 
		#print amx

		imprime_blocos(p, H, ota_img, w, inter)


                 
                core = {}
                delta = {}

		for i in range( 1, H-1 ):
			for j in range( 1, H-1 ):

				pode = 1
				for k in range ( i-1, i+2 ):
					for l in range ( j-1, j+2 ):
						if ( inter[k,l] == G ):
							pode = 0

				if ( pode ):

					soma = 0.0
					#print "--------------------"

					soma += delta_zuero( p[i-1, j-1], p[i, j-1] )
					soma += delta_zuero( p[i, j-1], p[i+1, j-1] )
					soma += delta_zuero( p[i+1, j-1], p[i+1, j] )
					soma += delta_zuero( p[i+1, j], p[i+1, j+1] )
					soma += delta_zuero( p[i+1, j+1], p[i, j+1] )
					soma += delta_zuero( p[i, j+1], p[i-1, j+1] )
					soma += delta_zuero( p[i-1, j+1], p[i-1, j] )
					soma += delta_zuero( p[i-1, j], p[i-1, j-1] )

					shue = (soma) / (2.0 * math.pi)
					#print 'delta_zuero:',  delta_zuero( p[i-1, j-1], p[i, j-1] )


					if (shue >= 0.45 and shue <= 0.55):
						print 'Core: ', i, j, soma, shue
                                                core[i] = j
						for k in range ( i*w, i*w+w ):
							for l in range ( j*w, j*w+w ):
								ota_img[k, l] = 0

					if (shue >= -0.55 and shue <= -0.3):
						print 'Delta: ', i, j, soma, shue
                                                delta[i] = j
						for k in range ( i*w, i*w+w ):
							for l in range ( j*w, j*w+w ):
								ota_img[k, l] = 0
					


		# *********************************** Contagem ************************************* #

                 
                print core
                print delta

                qtdCore = 0
                locCore = 0
                qtdDelta = 0
                locDelta = 0

                old_k = -99999
                for key, value in sorted(core.items()):
                    #print key, value
                    if ( abs(key - old_k) > 2): 
                        locCore = key
                        qtdCore += 1
                        old_k = key

                old_k = -99999
                for key, value in sorted(delta.items()):
                    #print key, value
                    if ( abs(key - old_k) > 2):     
                        locDelta = key
                        qtdDelta += 1
                        old_k = key

                print 'Core: ', qtdCore
                print 'Delta: ', qtdDelta
                all_core = np.append(all_core, qtdCore)
                all_delta = np.append(all_delta, qtdDelta)
               
                if ( locCore < locDelta ):
                    ang_core = np.append(ang_core, "left")
                    print "core = left"
                else: 
                    ang_core = np.append(ang_core, "right")
                    print "core = right"


		# *********************************** Image Binarization ************************************* #


                # Pegar imagem de novo...
		image = np.fromfile(image_path, dtype='uint8', sep="")
		image = image.reshape([SIZE, SIZE])
		A = image
                
                img_bin = np.zeros([SIZE, SIZE])
                
                img_blur = cv2.medianBlur(image,5)
                
                #hist,bins = np.histogram(img_blur.ravel(),256,[0,256])
                #print hist

                #img_bin = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                img_bin = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

                
                


		# *********************************** Plotagem ************************************* #
		'''
		plt.subplot(321),plt.imshow(B,cmap = 'gray')
		plt.title('Enhanced'), plt.xticks([]), plt.yticks([])
		plt.subplot(322),plt.imshow(blur,cmap = 'gray')
		plt.title('Blur'), plt.xticks([]), plt.yticks([])

		plt.subplot(323),plt.imshow(sobelx,cmap = 'gray')
		plt.title('Sobel x'), plt.xticks([]), plt.yticks([])
		plt.subplot(324),plt.imshow(sobely,cmap = 'gray')
		plt.title('Sobel y'), plt.xticks([]), plt.yticks([])

		plt.subplot(325),plt.imshow(ax,cmap = 'gray')
		plt.title('ax'), plt.xticks([]), plt.yticks([])
		plt.subplot(326),plt.imshow(ay,cmap = 'gray')
		plt.title('ay'), plt.xticks([]), plt.yticks([])

		plt.pause(_waitingtime)
                '''
		'''
                plt.subplot(121),plt.imshow(image,cmap = 'gray')
		plt.title('Original'), plt.xticks([]), plt.yticks([]) 
		plt.subplot(122),plt.imshow(img_bin,cmap = 'gray')
		plt.title('Threshold'), plt.xticks([]), plt.yticks([])
		#plt.subplot(111),plt.imshow(ota_img,cmap = 'gray')
		#plt.title(image_path), plt.xticks([]), plt.yticks([])

		plt.pause(_waitingtime)
		'''
                '''
		plt.subplot(121),plt.imshow(block,cmap = 'gray')
		plt.title('Block Orientation Images'), plt.xticks([]), plt.yticks([]) 
		plt.subplot(122),plt.imshow(ota_img,cmap = 'gray')
		plt.title('Teste de Resultado'), plt.xticks([]), plt.yticks([])

		mng = plt.get_current_fig_manager()
		mng.full_screen_toggle()
		plt.show()
		'''


print label
print all_core
print all_delta
 
voto = {}

all_core = all_core.astype(int)
all_delta = all_delta.astype(int)

old_label = "001"
save_c = np.array([])
save_d = np.array([])
save_l = np.array([])
for i in range( len(label) ):
    
    if ( label[i] != old_label or i == (len(label)-1) ):

        r_c = collections.Counter(save_c).most_common()[0][0]
        r_d = collections.Counter(save_d).most_common()[0][0]
        r_l = collections.Counter(save_l).most_common()[0][0]

        voto[label[i-1]] = [r_c, r_d, r_l]
        save_c = np.array([])
        save_d = np.array([])

    save_c = np.append(save_c, all_core[i])
    save_d = np.append(save_d, all_delta[i])
    save_l = np.append(save_l, ang_core[i])


print "Results: "        
for key, value in sorted(voto.items()):
    #print "Key: ", key, " - ", value
    if ( value[0] == 1 and value[1] == 1 ):
        print "Key: ", key, " - ", value, " =>", value[2], "loop"
    elif ( (value[0] <= 1 and value[1] == 0) or (value[0] == 0 and value[1] <= 1) ):
        print "Key: ", key, " - ", value, " => Arch"
    elif ( value[0] == 2 and value[1] <= 2 ):
        print "Key: ", key, " - ", value, " => Whorl"
    else:
        print "Key: ", key, " - ", value, " => Others..."


if (comp_path):
    print "Compare: "        

    files = [os.path.join(comp_path,f) for f in os.listdir(comp_path) if f.endswith('.lif') and os.path.isfile(os.path.join(comp_path,f))]
    files.sort()
    #print files
    for f in files:
        print f
        

        f_now = f.upper()
	now =  len(f_now.split('/')) - 1
	label_now =  f_now.split('/')[now]
	label_now = (label_now.split('R')[0]).split('F')[1]
        
        file_data = open(f).read()
        data = json.loads(file_data)
    
        core = 0
        delta = 0
        for es, s in enumerate(data["shapes"]):
            if ( s["label"] == "core" ):
                core += 1    
            else:     
                delta += 1

        print label_now, " = ",
        if ( core == voto[label_now][0] ):
            print "coreCerto, ",
        else:
            print "coreERRADO, ",
        
        if ( delta == voto[label_now][1] ):
            print "deltaCerto, "
        else:
            print "deltaERRADO, "
            
            #print es, s["label"]
            #for p in s["points"]:
            #    print p


#for i in range(len(voto)):
#    print i, voto[

