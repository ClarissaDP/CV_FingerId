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
from skimage.morphology import skeletonize
from skimage import img_as_bool


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

		for r in range(min_radius, max_radius):
		    xt = int(nj + r*coss)
		    yt = int(ni + r*sen)

		    block[yt,xt] = 0
		    #print i, j, xt, yt



def pode_agora(inter, ni,ei, nj,ej ):

    pode = 1
    if ( ei < (28) and ej < (28) ):
        ei += 1
        ej += 1

    for k in range ( ni, ei ):
        for l in range ( nj, ej ):
	    if ( inter[k,l] == G ):
		pode = 0

    return pode



def delta_zuero(a,b):
	diff = a - b
	
	if ( abs(diff) < math.pi/2 ):
		return diff
	if ( diff <= ((-1)*math.pi)/2 ):
		return math.pi + diff
	else: 
		return math.pi - diff



def smoothing(imaxe, t, fil):
     
    new  = np.zeros([SIZE, SIZE])

    rang = int( fil / 2 )
    for i in range( rang, (SIZE - rang) ):
        for j in range( rang, (SIZE - rang) ):
                
            wh = 0
            bl = 0

            #for k in range( (i-rang), (i+rang) ):
            #    for l in range( (j-rang), (j+rang) ):
            #        if ( imaxe[k,l] == 255 ):
            #            wh += 1
            #        elif ( imaxe[k,l] == 0 ):
            #            bl += 1
            
            wh = np.sum( (imaxe[(i-rang):(i+rang+1), (j-rang):(j+rang+1)]) == 255)
            #bl_n = (fil*fil-1) - wh
            bl = np.sum( (imaxe[(i-rang):(i+rang+1), (j-rang):(j+rang+1)]) == 0)

            if ( wh >= t ):
                new[i,j] = 255
            elif ( bl >= t ):
                new[i,j] = 0
            else:
                new[i,j] = imaxe[i,j]

    #return (new.astype(int))
    return new



def convert(imaxe):
    
    new  = np.zeros([SIZE, SIZE])
    
    for i in range (SIZE):
        for j in range (SIZE):
            if ( imaxe[i,j] != 0 ):
                new[i,j] = 255

    return new



def u_q(p1, p2):
    if(p1 != p2):
        return 1
    return 0
    #return (abs(p1-p2))



def nao_eh_borda( xi, xj, big_inter ):
    
    dis = 8
    yepa = 1
    for i in range( (xi-dis), (xi+dis) ):
        for j in range( (xj-dis), (xj+dis) ):
            if ( big_inter[i,j] == G ):
                yepa = 0
    return yepa



def elimina_minut( minut, big_inter ):
    new = np.array([])

    dis = 8

    l_m = len(minut)
    for i in range( 0, l_m, 3 ):
        #print minut[i], minut[i+1], minut[i+2]

        contin = 1

        if ( nao_eh_borda( int(minut[i+1]), int(minut[i+2]), big_inter ) ):
            if ( minut[i] == 1.0 ):
                k = i+3
                while ( k < l_m and contin ):
                    if ( abs(minut[i+1] - minut[k+1]) < dis and abs(minut[i+2] - minut[k+2]) < dis ):
                        contin = 0
                    k += 3        
        
            elif ( minut[i] == 3.0 ):
                k = i+3
                while ( k < l_m and contin ):
                    if ( minut[k] == 1.0 or minut[k] == 3.0 ):
                        if ( abs(minut[i+1] - minut[k+1]) < dis and abs(minut[i+2] - minut[k+2]) < dis ):
                            contin = 0
                    k += 3        
                 
            if (contin):
                new = np.append(new, minut[i:i+3])    

    return new





# ************************* Main *************************** #
if ( len(sys.argv) < 2 ):
	print "python fingerId_Classification.py imagesPath [plot]"
	exit(-1)

path = sys.argv[1]

plot = 0
if ( len(sys.argv) > 2 ):
    plot = sys.argv[2]

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
		big_inter = np.zeros([SIZE,SIZE])        
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
					    big_inter[k,l] = G 
				    inter[i,j] = G
	   
	  
		imprime_blocos(xy, H, block, w, inter)


		# ***************************** Singular Point Detection *********************************** #


		# Poincare'
		p = np.zeros([H, H])
		for i in range( 1, H-1 ):
			for j in range( 1, H-1 ):

				bloco_x = amx[(i-1):(i+1), (j-1):(j+1)]
				bloco_y = amy[(i-1):(i+1), (j-1):(j+1)]

				b_a = np.sum(bloco_x) + amx[i,j]
				b_b = np.sum(bloco_y) + amy[i,j]


				p[i,j] = (math.atan2(b_b, b_a) + np.pi) / 2.0
				

		ota_img = np.copy(img)

		print p.shape
		imprime_blocos(p, H, ota_img, w, inter)
                 
                core = {}
                delta = {}

		for i in range( 1, H-1 ):
			for j in range( 1, H-1 ):

                                if ( pode_agora( inter, (i-1), (i+2), (j-1), (j+2) ) ):

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
                
                #img_bin = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                img_bin = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

                
                
		# *********************************** Smoothing ************************************* #
                    
		#img_smooth = np.copy(img_bin)
                img_smooth = np.zeros([SIZE, SIZE])
                
                img_smooth = smoothing( img_bin, 18, 5 ) 
                img_smooth = smoothing( img_smooth, 5, 3 ) 



		# *********************************** Thinning ************************************* #
		
                skel = ~img_as_bool(img_smooth)
                img_skel = skeletonize(skel)                

                
		# *********************************** Minutiae Detection ************************************* #
                
                
                #img_minut = np.zeros([SIZE, SIZE])
		img_minut = np.copy(img_skel)
		p = convert(img_skel)
                H = 10

                minut = np.array([])

		for i in range( 1, SIZE-2 ):
			for j in range( 1, SIZE-2 ):

                                if ( pode_agora( inter, (i/H-1), (i/H+1), (j/H-1), (j/H+1) ) and p[i,j] == 255 ):

					soma = 0.0
					#print "--------------------"
                                            
					soma += u_q( p[i-1, j-1], p[i, j-1] )
					soma += u_q( p[i, j-1], p[i+1, j-1] )
					soma += u_q( p[i+1, j-1], p[i+1, j] )
					soma += u_q( p[i+1, j], p[i+1, j+1] )
					soma += u_q( p[i+1, j+1], p[i, j+1] )
					soma += u_q( p[i, j+1], p[i-1, j+1] )
					soma += u_q( p[i-1, j+1], p[i-1, j] )
					soma += u_q( p[i-1, j], p[i-1, j-1] )
                                        #print soma, i, j

                                        soma = soma / 2 
                                        if ( soma > 0 and soma < 5 and soma != 2):
                                            minut = np.append( minut, [soma, i, j] )

                                else:
                                    img_minut[i,j] = 0


                #print 'minut = ', minut

                minut = elimina_minut(minut, big_inter)
                print 'minut = ', minut

                for i in range( 0, len(minut), 3):
                    
                    ni = 1
                    ei = 2
                    #if ( minut[i] == 1.0 ):
                    #    ni = 2
                    #    ei = 3

                    for k in ( int(minut[i+1]) - ni, int(minut[(i+1)]) + ei ):
                        for l in ( int(minut[i+2]) - ni, int(minut[(i+2)]) + ei ):
                            img_minut[k,l] = G



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
                plt.subplot(221),plt.imshow(image,cmap = 'gray')
		plt.title(image_path), plt.xticks([]), plt.yticks([]) 
		#plt.subplot(222),plt.imshow(img_bin,cmap = 'gray')
		#plt.title('Threshold'), plt.xticks([]), plt.yticks([])
		plt.subplot(222),plt.imshow(img_smooth,cmap = 'gray')
		plt.title('Smoothing'), plt.xticks([]), plt.yticks([])
		plt.subplot(223),plt.imshow(img_skel,cmap = 'gray')
		plt.title('Skeleton'), plt.xticks([]), plt.yticks([])
		plt.subplot(224),plt.imshow(img_minut,cmap = 'gray')
		plt.title('Minutiae'), plt.xticks([]), plt.yticks([])

		plt.pause(_waitingtime)
		'''
		
                
                #plt.subplot(121),plt.imshow(p,cmap = 'gray')
		#plt.title('Skeleton'), plt.xticks([]), plt.yticks([])
		
                if (plot):
                    plt.subplot(111),plt.imshow(img_minut,cmap = 'gray')
		    plt.title('Minutiae'), plt.xticks([]), plt.yticks([])

		    plt.pause(_waitingtime)
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

