#some useful function ......
import os
import numpy
import math
import numpy as num
import matplotlib.pyplot as plt
from astropy.io import fits
from create_fits import create_fits
import astropy.units as u
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
from astropy.cosmology import FlatLambdaCDM
import wget
import scipy 
from scipy.stats import norm 
from scipy  import special 
from scipy import integrate 
from scipy.optimize import fsolve 

def read_compSpec():
	fp=open(r"C:\Users\32924\Desktop\vandenBerk2011.cds")
	lines=fp.readlines()
	fp.close()

	wave,flux,err=[],[],[]
	for line in lines:
		if '#' in line: continue 
		if line.strip()=='': continue 
		wave.append(float(line.split()[0]))
		flux.append(float(line.split()[1]))
		err.append(float(line.split()[2]))
	return num.array(wave),num.array(flux),num.array(err)


def set_axis(ax,labelsize=None,linewidth=None,direction=None,major_length=None,major_width=None,minor_length=None,minor_width=None):
	if not  labelsize==None:
		ax.tick_params(labelsize=labelsize)
	if not  direction==None:
		ax.tick_params(which='both',direction=direction)
	if not major_length==None: 
		ax.tick_params(which='major',length=major_length)
	if not major_width==None:
		ax.tick_params(which='major',width=major_width) 
	if not minor_length==None:
		ax.tick_params(which='minor',length=minor_length) 
	if not minor_width==None: 
		ax.tick_params(which='minor',width=minor_width) 
	
	if not linewidth==None:
		ax.spines['bottom'].set_linewidth(linewidth)
		ax.spines['left'].set_linewidth(linewidth)
		ax.spines['right'].set_linewidth(linewidth)
		ax.spines['top'].set_linewidth(linewidth)
def set_ticks(axis,major=None,minor=None,both=True):
	if not major==None:
		majorlocator=MultipleLocator(major)
		axis.set_major_locator(majorlocator)
		# if (length is not None):
		# 	axis.tick_params(which='major',length=length)
		# if (width is not None):
		# 	axis.tick_params(which='major',width=width)
		# if (labelsize is not None):
		# 	axis.tick_params(which='major',labelsize=labelsize)
	if not minor==None:
		minorlocator=MultipleLocator(minor)
		axis.set_minor_locator(minorlocator)
		# if (length is not None):
		# 	axis.tick_params(which='minor',length=length)
		# if (width is not None):
		# 	axis.tick_params(which='minor',width=width)
		# if (labelsize is not None):
		# 	axis.tick_params(which='minor',labelsize=labelsize)
	if both: 
		axis.set_ticks_position('both')
#smooth the spectra
def smooth_convol(x,window_len=11,window='hanning'):
	"""smooth the data using a window with requested size.

	This method is based on the convolution of a scaled window with the signal.
	The signal is prepared by introducing reflected copies of the signal 
	(with the window size) in both ends so that transient parts are minimized
	in the begining and end part of the output signal.

	input:
		x: the input signal 
		window_len: the dimension of the smoothing window; should be an odd integer
		window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
		flat window will produce a moving average smoothing.
	output:
		the smoothed signal

	example:

	t=linspace(-2,2,0.1)
	x=sin(t)+randn(len(t))*0.1
	y=smooth(x)

	see also: 

	numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
	scipy.signal.lfilter

	TODO: the window parameter could be the window itself if an array instead of a string
	NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
	"""
	if x.ndim != 1:
		raise ValueError("smooth only accepts 1 dimension arrays.")

	if x.size < window_len:
		raise ValueError("Input vector needs to be bigger than window size.")


	if window_len<3:
		return x


	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


	s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
	#print(len(s))
	if window == 'flat': #moving average
		w=numpy.ones(window_len,'d')
	else:
		w=eval('numpy.'+window+'(window_len)')

	y=numpy.convolve(w/w.sum(),s,mode='same') #or valid?
	return y
#smooth the spectra using simply median method
def smooth(wave,flux,N,show=True):
#smooth a spectra
	length=len(flux)
	fluxsm=[]
	for i in range(0,length):
		if i>=N and i<=length-N-1:
			fluxsm.append(num.median(flux[i-N:i+N]))
	wavesm=wave[N:length-N]
	flux=flux[N:length-N]

	if show:
		pl1,pl2,pl3,=plt.plot(wavesm,fluxsm,'r',wavesm,flux,'b',wavesm,flux-fluxsm,'g',alpha=0.7)
		plt.legend(handles=[pl1,pl2,pl3],labels=['smooth','original','residual'])
		plt.show()
	return num.array(wavesm),num.array(flux),num.array(fluxsm)
#evaluate the flux error...
def produce_err(path,mode='smooth'):
	hdu=fits.open(path)
	wave=hdu[1].data['wavelength']
	flux=hdu[1].data['flux']
	wave,flux,flux_smooth=smooth(wave,flux,7)
	plt.plot(wave,flux,wave,flux_smooth)
	plt.plot(wave,abs(flux-flux_smooth))
	plt.show()
	print('create reasulting files....')
	datas={0:None,1:{'wavelength':{'data':wave,'fmt':'D','unit':'Angstrom'},\
					'flux':{'data':flux,'fmt':'D','unit':'1e-17erg/s/cm^2/A'},\
					'err':{'data':abs(flux-flux_smooth),'fmt':'D','unit':'1e-17erg/s/cm^2/A'}}}
	params={0:{},1:{}}
	#hdu.close()
	#os.remove(path)
	create_fits(datas,params,path.replace('.fits','1.fits'))
	return flux_smooth

# purtrubing the flux within the error according to the normal ditribution or others
def MonteC(flux_dbsp,err_dbsp,model='normal'):
	flux=num.ones_like(flux_dbsp)
	if model=='normal':
		for i in range(0,len(flux_dbsp)):
			flux[i]=num.random.normal(flux_dbsp[i],err_dbsp[i])
	return flux
#find the possible peak in the spectra, and give their position
def findpeak(wave,flux,n,Npoint,Nsigma):
	length=len(flux)
	Npoint=200.
	N=int(round(length/Npoint))
	std=[0]*N # NOTE would this raise any question ?
	for i in range(1,N):
		segeflux=flux[(i-1)*200:i*200]
		std[i-1]=num.std(segeflux)
	segeflux=flux[(N-1)*200:length-1] #?
	std[N-1]=num.std(segeflux)
	fluxup=[]
	fluxlow=[]
	for i in range(0,length):
		if i>=n and i<=length-n-1:
			median=num.median(flux[i-n:i+n])
			N1=math.floor(i/Npoint)
			if N1>N-1:
				N1=N-1
			N1=int(N1)
			fluxup.append(median+Nsigma*std[N1])
			fluxlow.append(median-Nsigma*std[N1])
	wavesm=wave[n:length-n]
	ind=(flux[n:length-n]<fluxlow)|(flux[n:length-n]>fluxup)
	ind=num.insert(ind,0,[True]*n)
	ind=num.append(ind,[True]*n)
	return ind
#.......
def aline(w1,w2,*params):
	I1=0
	while w1[I1]<w2[0]:
		I1=I1+1
	I2=len(w1)-1
	while w1[I2]>w2[len(w2)-1]:
		I2=I2-1
	cutparams=[0]*(len(params)+1)
	cutparams[0]=w1[I1:I2+1]
	for  i in range(0,len(params)):
		cutparams[i+1]=params[i][I1:I2+1]
	return cutparams	
#interploate w2 to w1, and the err was given by error protagation
def Langrange(w1,w2,f2,f2err):
# interp1d w2 to w1 
#	print 'len(ferr)==%s'%len(f2err)
	l1=len(w1)
	l2=len(w2)
	i1=0
	i2=0
	I=l1-1
	f1=[]
	f1err=[]
	while w1[i1]<w2[0]:
		i1=i1+1
	for i in range(0,i1):
		f=f2[0]+(w1[i]-w2[0])/(w2[1]-w2[0])*(f2[1]-f2[0])
		ferr=(f2err[0]**2+((w1[i]-w2[0])/(w2[1]-w2[0]))**2*(f2err[1]**2+f2err[0]**2))**0.5
		f1.append(f)
		f1err.append(ferr)
	while w1[I]>w2[l2-1]:
		I=I-1
	for i in range(i1,I+1):
		while w1[i]<w2[i2] or w1[i]>w2[i2+1]:
			i2=i2+1
#			print w1[i],w1[i-1],l1,l2,i2,w2[i2],'\n'
		f=f2[i2]+(w1[i]-w2[i2])/(w2[i2+1]-w2[i2])*(f2[i2+1]-f2[i2])
		ferr=(f2err[i2]**2+((w1[i]-w2[i2])/(w2[i2+1]-w2[i2]))**2*(f2err[i2+1]**2+f2err[i2]**2))**0.5
		f1.append(f)
		f1err.append(ferr)

	for i in range(I+1,l1):
		f=f2[l2-1]+(w1[i]-w2[l2-1])/(w2[l2-2]-w2[l2-1])*(f2[l2-2]-f2[l2-1])

		ferr=(f2err[l2-1]**2+((w1[i]-w2[l2-1])/(w2[l2-2]-w2[l2-1]))**2*(f2err[l2-2]**2+f2err[l2-1]**2))**0.5
		f1.append(f)
		f1err.append(ferr)
	f1=num.array(f1)
	f1err=num.array(f1err)
	return f1,f1err
#some useful lines' position
def indline(wave,name):
	if name=='balmer':
		N_angstrom_masked=20
		ind =   ((wave > 4341 - N_angstrom_masked) & (wave < 4341 + N_angstrom_masked)) | \
			((wave > 4861 - N_angstrom_masked) & (wave < 4861 + N_angstrom_masked)) | \
			((wave > 6564 - 50) & (wave < 6564 + 50)) 
		return ind
	if name=='Coronal':
		N_angstrom_masked=7.5
		ind =   ((wave > 3759 - N_angstrom_masked) & (wave< 3759 + N_angstrom_masked)) | \
			((wave > 3869 - N_angstrom_masked) & (wave< 3869 + N_angstrom_masked)) | \
			((wave > 4414 - N_angstrom_masked) & (wave< 4414 + N_angstrom_masked)) | \
			((wave > 4686 - N_angstrom_masked) & (wave< 4686 + N_angstrom_masked)) | \
			((wave > 5160 - N_angstrom_masked) & (wave< 5160 + N_angstrom_masked)) | \
			((wave > 5304 - N_angstrom_masked) & (wave< 5304 + N_angstrom_masked)) | \
			((wave > 5722 - N_angstrom_masked) & (wave< 5722 + N_angstrom_masked)) | \
			((wave > 6088 - N_angstrom_masked) & (wave< 6088 + N_angstrom_masked)) | \
			((wave > 6376 - N_angstrom_masked) & (wave< 6376 + N_angstrom_masked)) | \
			((wave > 7612 - N_angstrom_masked) & (wave< 7612 + N_angstrom_masked)) | \
			((wave > 7894 - N_angstrom_masked) & (wave< 7894 + N_angstrom_masked))
		return ind
	if name=='s2+o3':
		ind=    ((wave>6710) & (wave<6740)) | ((wave>4940) & (wave<5027))
		return ind	
	if name=='s2':
		ind=    ((wave>6710) & (wave<6740))
		return ind
	if name=='oxygen':
		N_angstrom_masked=20
		ind=    ((wave > 3728 - N_angstrom_masked) & (wave< 3728 + N_angstrom_masked)) | \
			((wave > 4960 - N_angstrom_masked) & (wave< 4960 + N_angstrom_masked)) | \
			((wave > 5007 - N_angstrom_masked) & (wave< 5007 + N_angstrom_masked))
		return ind
	if name=='absorption':
		ind=    ((wave >3770) & (wave <4050)) | \
			((wave >5100) & (wave <5300)) | \
			((wave >5860) & (wave <5910)) | \
			((wave >8450) & (wave <8550))
#			((wave >6710) & (wave <6750)) | \
		return ind
	if name=='telluric':
		# noting: the wavelength here is not the restframe's 
		# ind=((wave >7600) & (wave <7630)) | \
		# 	((wave >6860) & (wave <6890)) | \
		# 	((wave >7170) & (wave <7350))
		ind=((wave >7580) & (wave <7750)) | \
			((wave >6860) & (wave <6960)) | \
			((wave >7160) & (wave <7340)) | \
			((wave >8150) & (wave <8250)) 
		return ind

#     The following data was got from 
#        ## Mask telluric region in the optical
#        tell_opt = np.any([((wave_star >= 6270.00) & (wave_star <= 6290.00)), # H2O
#                       ((wave_star >= 6850.00) & (wave_star <= 6960.00)), #O2 telluric band
##                       ((wave_star >= 7580.00) & (wave_star <= 7750.00)), #O2 telluric band
#                       ((wave_star >= 7160.00) & (wave_star <= 7340.00)), #H2O
#                       ((wave_star >= 8150.00) & (wave_star <= 8250.00))],axis=0) #H2O
	if name=='error':
		ind=    ((wave >5641.6) & (wave <5665.05)) #?
		return ind
def lname2print(lname):
	if lname=='halpha': return r'$H_\alpha$'
	if lname=='hbeta': return r'$H_\beta$'
	if lname=='O3': return r'$[OIII]\lambda$'
	if lname=='sii': return r'$[SII]\lambda$'
	if lname=='HeII': return r'$HeII\lambda4686$'
	if lname=='NIII': return r'$NIII\lambda4640$'
	if lname=='FeVII6088': return r'$[FeVII]\lambda6088$'
	if lname=='FeVII5722': return r'$[FeVII]\lambda5722$'
	if lname=='FeX': return [r'$[OI]\lambda6301$',r'$[OI]\lambda6364$','FeX']
	if lname=='FeXIV5304': return r'$[FeXIV]\lambda5304$'
	if lname=='FeXI7894': return r'$[FeXI]\lambda7894$'
	if lname=='FeV4071': return r'$[FeV]\lambda4071$'
	if lname=='FeVII5304': return r'$[FeVII]\lambda5304$'
	if lname=='FeVII3759': return r'$[FeVII]\lambda3759$'
def lname2wave(lname):
	if lname=='HeII': return 4686
	if lname=='NIII': return 4640
	if lname=='FeVII6088': return 6088
	if lname=='FeVII5722': return 5722
	if lname=='FeX': return 6376
	if lname=='FeXIV5304': return 5304
	if lname=='FeXI7894': return 7894
def plot_spectra(datas,redshift=None,redshift_correct=False,signlines=False):
	wave,flux=datas['wave'],datas['flux']
	if redshift_correct:
		wave=wave/(1+redshift)
def plot_lines(ax,redshift=0,linews=None,coronal_lines=None,balmer_lines=None):
	#ax is the axis to sign the lines 
	if not coronal_lines==None:
		for line in coronal_lines:
			pass
	if not linews==None:
		for linew in linews:
			pass
	if not balmer_lines==None:
		for line in balmer_lines:
			pass
	return ':)'

#merging the two ends of the spectra
def merge(path_blue,path_red,Nmin,lowcut,drop_tail=None,scale=True,model='multiplitive',path_savefig=None,path_savefile=None,scaleto='red',plot=True,show=True,savefile=True,fmt={}):
	#give the data format of the hdu[1].data
	if fmt=={}:
		fmt={'wave':'wavelength','flux':'flux','err':'err','var':'var','ivar':'ivar','unit':'erg/s'}
	#print(fmt)
	#considering that only 1 of the two ends was existed
	if not (os.path.exists(path_blue)&os.path.exists(path_red)):
		if not os.path.exists(path_blue):
			path=path_red
			exists='red'
		else: 
			path=path_blue
			exists='blue'
		hdu=fits.open(path)
		hdu[1].header['merge_status']='only %s end spectrum exists!!!'%exists	
		wave=hdu[1].data[fmt['wave']]
		flux=hdu[1].data[fmt['flux']]
		if fmt['unit']=='1e-17erg/s':
			flux=flux*1e-17
			err=err*1e-17
		elif fmt['unit']=='erg/s':
			flux=flux
		else:
			raise Exception('No macth units: %s'%fmt['unit'])
		ind_nan=~num.isnan(flux)
		wave,flux=wave[ind_nan],flux[ind_nan]
		if exists=='blue':
			hdu[1].header['wave_point']=wave[len(wave)-1]
			hdu[1].header['wave_connect']=wave[len(wave)-1]
		else:
			hdu[1].header['wave_point']=wave[0]
			hdu[1].header['wave_connect']=wave[0]
		hdu[1].header['ratio']=-1
		hdu[1].header['ratio_err']=-1
		if plot:
			plt.plot(wave,flux,'g',alpha=0.7)
			y=num.arange(num.min(flux),num.max(flux),(num.max(flux)-num.min(flux))/30)
			plt.plot([hdu[1].header['wave_connect']]*len(y),y,'k--',alpha=0.7)
	#		plt.show()
			if not path_savefig==None:
				if os.path.exists(path_savefig):
					os.remove(path_savefig)
				plt.savefig(path_savefig)
				plt.close()
		if savefile:
			if os.path.exists(path_savefile):
				os.remove(path_savefile)
			hdu.writeto(path_savefile)
		return wave,flux,-1
	#if the resoltuion of the two ends of the spectra was very different, the points number Nmin should be changed to wavelength range
	rhdu=fits.open(path_red)
	rhdu=rhdu[1]
	bhdu=fits.open(path_blue)
	bhdu=bhdu[1]
	bwave=bhdu.data[fmt['wave']]
	bflux=bhdu.data[fmt['flux']]
	try:
		berr=bhdu.data[fmt['err']]
	except:
		try:berr=bhdu.data[fmt['var']]**0.5
		except:berr=bhdu.data[fmt['ivar']]**-0.5
	if fmt['unit']=='1e-17erg/s':
		bflux=bflux*1e-17
		berr=berr*1e-17
	elif fmt['unit']=='erg/s':
		bflux=bflux
	else:
		raise Exception('No macth units: %s'%fmt['unit'])
	ind_nan=~(num.isnan(bflux) | num.isnan(berr))
	bwave,bflux,berr=bwave[ind_nan],bflux[ind_nan],berr[ind_nan]
	ind_inf=~(num.isinf(bflux) | num.isinf(berr))
	bwave,bflux,berr=bwave[ind_inf],bflux[ind_inf],berr[ind_inf]
	rwave=rhdu.data[fmt['wave']]
	rflux=rhdu.data[fmt['flux']]
	try:
		rerr=rhdu.data[fmt['err']]
	except:
		try:rerr=rhdu.data[fmt['var']]**0.5
		except: rerr=rhdu.data[fmt['ivar']]**-0.5
	if fmt['unit']=='1e-17erg/s':
		rflux=rflux*1e-17
		rerr=rerr*1e-17
	elif fmt['unit']=='erg/s':
		rflux=rflux
	else:
		raise Exception('No macth units: %s'%fmt['unit'])
	ind_nan=~(num.isnan(rflux)| num.isnan(rerr))
	rwave,rflux,rerr=rwave[ind_nan],rflux[ind_nan],rerr[ind_nan]
	ind_inf=~(num.isinf(rflux)| num.isinf(rerr))
	rwave,rflux,rerr=rwave[ind_inf],rflux[ind_inf],rerr[ind_inf]
	if not drop_tail==None: #if the tail of one of the two spectra was two bad, we remove them
		if 'blue' in drop_tail.keys():
			ind=bwave<0 
			for region in drop_tail['blue']: 
				ind= ind| ( (bwave>region[0]) & (bwave<region[1]) )  
	#		ind=(bwave>drop_tail['blue'][0]) & (bwave<drop_tail['blue'][1]) 
	#		ind=ind | (bwave<3980) # only for the spectrum J1105-200213
			bwave,bflux,berr=bwave[ind==False],bflux[ind==False],berr[ind==False]
		if 'red' in drop_tail.keys():
			ind=rwave<0 
			for region in drop_tail['red']: 
				ind= ind|( (rwave>region[0]) & (rwave<region[1]) ) 
		#	ind=(rwave>drop_tail['red'][0]) & (rwave<drop_tail['red'][1])
			rwave,rflux,rerr=rwave[ind==False],rflux[ind==False],rerr[ind==False]

	bwmax=bwave[len(bwave)-1]
	rwmin=rwave[0]

	if bwmax<rwmin:
		merge_status='the middle is disconnection!!!'
		bdw=bwave[len(bwave)-Nmin:len(bwave)]
		bdf=bflux[len(bflux)-Nmin:len(bflux)]
		bde=berr[len(bwave)-Nmin:len(bwave)]
		rdw=rwave[0:Nmin]
		rdf=rflux[0:Nmin]
		rde=rerr[0:Nmin]
		wave_point=-1
		wave_connect=-1
	else: 
		merge_status='normal'
		wave_point=(bwmax+rwmin)/2
		overlap_bwave=bwave[(bwave>rwmin) & (bwave<bwmax)]
		overlap_bflux=bflux[(bwave>rwmin) & (bwave<bwmax)]
		overlap_berr=berr[(bwave>rwmin) & (bwave<bwmax)]
		overlap_rwave=rwave[(rwave>rwmin) & (rwave<bwmax)]
		overlap_rflux=rflux[(rwave>rwmin) & (rwave<bwmax)]
		overlap_rerr=rerr[(rwave>rwmin) & (rwave<bwmax)]	
		#assume the wavelength interval was similar between the red and blue ends spectra and so that the same number of data points can represent the same wavelength range
		if len(overlap_bwave)<Nmin:
			cut_bwave=bwave[bwave<wave_point]
			#cut_rwave=rwave[rwave>wave_point]
			overlap_bflux=bflux[bwave<wave_point][len(cut_bwave)-Nmin:len(cut_bwave)]
			overlap_berr=berr[bwave<wave_point][len(cut_bwave)-Nmin:len(cut_bwave)]
			overlap_rflux=rflux[rwave>wave_point][0:Nmin]
			overlap_rerr=rerr[rwave>wave_point][0:Nmin]
		bdw,bdf,bde=overlap_bwave,overlap_bflux,overlap_berr
		rdw,rdf,rde=overlap_rwave,overlap_rflux,overlap_rerr

			
	if scaleto=='red':
		if model=='multiplitive':
			ratio=( sum(rdf)/len(rdf) ) / (sum(bdf)/len(bdf))  # the lengths of the variables overlap_bwave and overlap_rwaves can be different and so that the lengths of 'rdf' and 'bdf'
			ratio_err=(sum(rde**2)/sum(rdf)**2+sum(bde**2)/sum(bdf)**2)**0.5*ratio
	#		ratio_SN=ratio/ratio_err

	#       
	#		lowcut=10
			wave_connect=rwave[lowcut] # .....
			wave=num.hstack([bwave[bwave<wave_connect],rwave[lowcut:] ])
			if scale:
				flux=num.hstack([bflux[bwave<wave_connect]*ratio,rflux[lowcut:] ])
			# consider the err of connect the two ends spectra
				oerr=num.mean(berr)
				berr=(berr**2/bflux**2+ratio_err**2/ratio**2)**0.5*bflux*ratio
				print(num.mean(berr)/ratio/oerr)
				err=num.hstack([berr[bwave<wave_connect],rerr[lowcut:] ])
			else:
				flux=num.hstack([bflux[bwave<wave_connect],rflux[lowcut:] ])
				err=num.hstack([berr[bwave<wave_connect],rerr[lowcut:] ])
		elif model=='additive':
			deference= sum(rdf)/len(rdf)-sum(bdf)/len(bdf)
			wave_connect=rwave[lowcut]
			wave=num.hstack([bwave[bwave<wave_connect],rwave[lowcut:] ])
			if scale:
				flux=num.hstack([bflux[bwave<wave_connect]+deference,rflux[lowcut:]])
				err=num.hstack([berr[bwave<wave_connect],rerr[lowcut:]])
			else:
				flux=num.hstack([bflux[bwave<wave_connect],rflux[lowcut:] ])
				err=num.hstack([berr[bwave<wave_connect],rerr[lowcut:] ])
		else:
			raise Exception('No such split joint model')
	elif scaleto=='blue':
		if model=='additive':
			deference= sum(rdf)/len(rdf)-sum(bdf)/len(bdf)
			wave_connect=rwave[lowcut]
			wave=num.hstack([bwave[bwave<wave_connect],rwave[lowcut:] ])
			if scale:
				flux=num.hstack([bflux[bwave<wave_connect],rflux[lowcut:]-deference])
				err=num.hstack([berr[bwave<wave_connect],rerr[lowcut:]-deference])
			else:
				flux=num.hstack([bflux[bwave<wave_connect],rflux[lowcut:] ])
				err=num.hstack([berr[bwave<wave_connect],rerr[lowcut:] ])			
		elif model=='multiplitive':
			raise Exception('Please add this part!!!!')

	if plot:
		plt.plot(bwave,bflux,'b',rwave,rflux,'r',wave,flux,'g',alpha=0.7)
		print(num.min(flux),num.max(flux))
		y=num.arange(num.min(flux),num.max(flux),(num.max(flux)-num.min(flux))/30.0)
		plt.plot([wave_connect]*len(y),y,'k--',alpha=0.7)
	#	plt.show()
		if not path_savefig==None:
			if os.path.exists(path_savefig):
				os.remove(path_savefig)
			plt.savefig(path_savefig)
		if show:
			plt.show()
		plt.close()
	if savefile:
		print('Creating the resultting files...')
		datas={0:None,1:{'wave_dbsp':{'data':wave,'fmt':'D','unit':'Angstrom'},\
						'flux_dbsp':{'data':flux,'fmt':'D','unit':'erg/s/cm^2/A'},\
						'err_dbsp':{'data':err,'fmt':'D','unit':'erg/s/cm^2/A'}}}
		if model=='multiplitive':
			params={0:{},1:{'wave_point':wave_point,'wave_connect':wave_connect,'ratio':ratio,'ratio_err':ratio_err,'merge_status':merge_status}}
			create_fits(datas,params,path_savefile)
			return wave,flux,ratio
		elif model=='additive':
			params={0:{},1:{'wave_point':wave_point,'wave_connect':wave_connect,'difference':deference,'merge_status':merge_status}}
			create_fits(datas,params,path_savefile)
			return wave,flux,deference
		else: 
			raise Exception('No such split joint model!!!')
			return ':)'

def Liangji(value,value_err):
	if (value==0)&(value_err==0):
		return 0
	if not value==0:
		zvalue=value
	else:
		zvalue=value_err
	i=0
	while 1:
#		print((value/float(10.**i)))
		if ((zvalue/float(10.**i))<10)&((zvalue/float(10.**i))>=1):
			break
		i=i+1
#		print(i)
	return i
def valid_number(value,err):
	if value==0:
		return value,round(err,1)
	nerr=err/1.
	for i in [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000,1000000]:
		if (nerr>i)&(nerr<i*10):
			if i>=1:
	#			print 'hello'
				value=float(int(round(value/float(i))*i))
				err=float(int(round(err/float(i))*i))

				return value,err
			else:
	#			print 'hello'
				j=0
				while 1:
					j=j+1
					if 10**j*i==1:
						break
				value=round(value,int(j))
				err=round(err,int(j))
				return value,err
	print('i Out of range')
	return ':('
								
def mjd2ut(mjd):
	Y=int((mjd-15078.2)/365.25)
	M=int((mjd-14956.1-int(Y*365.25))/30.6001)
	D=mjd-14956-int(Y*365.25)-int(M*30.6001)
	if (M==14)|(M==15):
		K=1
	else:
		K=0
	Y=Y+K
	M=M-1-K*12
	
	Y=Y+1900
	return '%d'%Y+'%02d'%M+'%02d'%D
def ut2mjd(ut):
	Y=float('20'+ut[0:2])
	Y=Y-1900
	M=float(ut[2:4])
	D=float(ut[4:6])
	
	if (M==1)|(M==2):
		L=1
	else:
		L=0

	MJD=14956+D+int((Y-L)*365.25)+int((M+1+L*12)*30.6001)
	return MJD
def LickHd(wave,flux,err=None,MC=False,MC_number=1000):

	#ind =(wave>4041.60)&(wave<4161.60)
	ind1=(wave>4083.50)&(wave<4122.25)
	ind2=(wave>4041.60)&(wave<4079.75)
	ind3=(wave>4128.50)&(wave<4161.00)

	if MC==True:
		if err is None: raise Exception('err is neccessary when to do MC')
	else: MC_number=1
	EWs,flux0=[],flux.copy()
	for i in range(MC_number):
		if MC:
			flux=MonteC(flux0,err) #You may need to improve this part to speed faster 

		#pseudo continuum, average flux in region ind1 and ind2, and a linear point joined these two average points 
		conti1=sum( [flux[ind2][x]*(wave[ind2][x+1]-wave[ind2][x]) for x in range(0,len(wave[ind2])-1)]) / (max(wave[ind2])-wave[ind2][0])
		conti2=sum( [flux[ind3][x]*(wave[ind3][x+1]-wave[ind3][x]) for x in range(0,len(wave[ind3])-1)]) / (max(wave[ind3])-wave[ind3][0])
	#	print(conti1,sum(flux[ind2])/len(flux[ind2]),conti2,sum(flux[ind3])/len(flux[ind3]))

		w1,w2=4060.675,4144.75
		conti=conti1+(conti2-conti1)/(w2-w1)*(wave[ind1]-w1)
			
		EW=sum([ (1-flux[ind1][x]/conti[x])*(wave[ind1][x+1]-wave[ind1][x]) for x in range(0,len(flux[ind1])-1)])
		EWs.append(EW)
	EW=num.mean(EWs)
	EW_err=num.std(EWs,ddof=1)

	return EW,EW_err

def DL(redshift):
	cosmo=FlatLambdaCDM(H0=70*u.km/u.s/u.Mpc,Om0=0.3,Tcmb0=2.725*u.K)
	dl=cosmo.luminosity_distance(redshift)
	return dl
def Mag2Abs(mag, redshift): 
	dl=DL(redshift)
	dl=dl.to(u.pc).value
	mag_abs=mag-5*num.log10(dl/10)
	return mag_abs  

def flux2L(flux,redshift, H0_used=70):
	#Hubble constant
	cosmo=FlatLambdaCDM(H0=H0_used*u.km/u.s/u.Mpc,Om0=0.3,Tcmb0=2.725*u.K)
	dl=cosmo.luminosity_distance(redshift)
	luminosity=flux*4*num.pi*dl**2
	luminosity=luminosity.to('erg/s').value #the unit is 'erg/s'
	return luminosity
def L2flux(lumi, redshift): 
	cosmo=FlatLambdaCDM(H0=70*u.km/u.s/u.Mpc,Om0=0.3,Tcmb0=2.725*u.K)
	dl=cosmo.luminosity_distance(redshift)
	flux=lumi/ (4*num.pi*dl**2 )
	flux=flux.to('erg/s/cm^2').value
	return flux 
def Mag2Lumi(mag, fz, lam, redshift=None, dist=None): 
	if (redshift is None) & (dist is None): 
		raise Exception('Redshift or distance must be provided') 
	
	f_lam=10**(-0.4*mag)*fz 
	if redshift is not None:
		Lumi= flux2L(lam*f_lam,redshift)
	else: 
		# print(mag.__class__)
		# print(fz)
		# print(f_lam)
		# print(lam*f_lam)
		Lumi= lam*f_lam *4*num.pi*dist**2
		Lumi= Lumi.to('erg/s').value #the unit is 'erg/s'
	return Lumi 

def WISE_Lumi(mag, band='W1', redshift=None, dist=None, mag_system='vega'): 
	#NOTE the default mag_system is 'vega' 
	if mag_system=='vega': 
		if band=='W1': fz=  8.18e-12 *u.Unit('erg/s/cm^2/AA') ; lam=33526.00*u.Unit('AA')
		if band=='W2': fz=  2.42e-12 *u.Unit('erg/s/cm^2/AA') ; lam=46028.00*u.AA 
	elif mag_system=='AB': 
		if band=='W1': fz=  8.18e-12 *u.Unit('erg/s/cm^2/AA') ; lam=33526.00*u.Unit('AA')
		if band=='W2': fz=  2.42e-12 *u.Unit('erg/s/cm^2/AA') ; lam=46028.00*u.AA 
	else: raise Exception('Please choose the mag_system from vega and AB')


	return Mag2Lumi(mag,fz,lam,dist=dist,redshift=redshift)

def bolL(L,mode='Ha'):
#	raise Exception('Please make sure the L here is in fact lam*L_lam or just L_lam?')
	return 8.1*5100*L
def BHmasa(L,FWHM,lerr=0,fwhm_err=0,line='ha',output='normal'):
	if line=='ha':
		Mbh=2e6*(L/1e42)**0.55*(FWHM/1e3)**2.06 #unit Msun
		Mbh_up=2.4e6*(L/1e42)**0.57*(FWHM/1e3)**2.12
		Mbh_low=1.7e6*(L/1e42)**0.53*(FWHM/1e3)**2.0
		Mbh_err=2e6*((L+lerr)/1e42)**0.55*((FWHM+fwhm_err)/1e3)**2.06-Mbh # the error here was not accurate,and actually we should take the MC simulations
	if output=='normal':
		return Mbh,Mbh_up,Mbh_low,Mbh_err
	elif output=='log10':
		return num.log10(Mbh),num.log10(Mbh_up),num.log10(Mbh_low),num.log10(Mbh_err)
	# See the paper Greene 2005, the correlation above is derive from (1) and (3), and the R-L relation, and virial formula, 
def MBH_CIV(L1350, FWHM):
	# using the results from Vestergaard&Peterson 2006(https://iopscience.iop.org/article/10.1086/500572/pdf) 
	# using the equation (7) of the above paper 
	lgMBH=num.log10( (FWHM/1000)**2 * (L1350/1e44)**0.53  ) +6.66
	raise Exception('PLease change the error output if considering the error of FWHM')
	return lgMBH, 0.36 #NOTE here 0.36 is the The sample standard deviation of the weighted average zero point offset
def lamLangstrom(L,L_err,line='ha'):
	if line=='ha':
		L_ang=(L/5.25e42)**(1/1.157)*1e44
		L_ang_err=((L+L_err)/5.25e42)**(1/1.157)*1e44-L_ang
	#NOTE the L in fact is lam*L_lam
	return L_ang,L_ang_err
def readname(namef):
#read name into a dictionary
	fp=open(namef)
	lines=fp.readlines()
	fp.close()
	key=lines[0].split('#')[1].strip()
	fnames={key:[]}
	for line in lines:
		if line.strip()=='':
			continue
		if line[0]=='#':
			if line.split('#')[1].strip() not in fnames.keys():
				key=line.split('#')[1].strip()
				fnames[key]=[]
			else:
				key=line.split('#')[1].strip()
				print('Same key?')
			continue
		fnames[key].append(line.strip())
	return fnames
def readfname(path):
#read the fname into a list
	fp=open(path)
	lines=fp.readlines()
	fp.close()

	fnames=[]
	for line in lines:
		if line.strip()=='':
			continue
		fnames.append(line.strip())
	return fnames
def readid():
	Path=r'D:\new_infrared\workspace\ALL'
	fp=open(r'D:\new_infrared\output\plateid.txt','w')
	fnames=os.listdir(Path)
	for fname in fnames:
		path_sdss=os.path.join(Path,fname,'%s_sdss.fits'%fname[4:9])
		if not os.path.exists(path_sdss):
			continue
		hdu=fits.open(path_sdss)
		plateid=hdu[0].header['Plateid']
		mjd=hdu[0].header['MJD']
		fiberid=hdu[0].header['fiberid']

		fp.write('%s %s %s %s\n'%(fname,plateid,mjd,fiberid))
	fp.close()
	return ':)'
def readlinefits(path,params):
	return ':)'
def plot_profile(x,y,col='k',linestyle='.',linewidth=None,alpha=None):
	for i in range(0,len(x)-1):
		pli,=plt.plot([x[i],x[i+1]],[y[i],y[i]],col)
	return pli

def cwe(a,b,a_err,b_err,operator):
	if operator=='/':
		x=a/b
		x_err=((a_err/a)**2+(b_err/b)**2)**0.5*x
		return x,x_err
	if operator=='*':
		x=a*b
		x_err=((a_err/a)**2+(b_err/b)**2)**0.5*x
		return x,x_err
	if operator=='+':
		x=a+b
		x_err=(a_err**2+b_err**2)**0.5
		return x,x_err
	if operator=='-':
		x=a-b
		x_err=(a_err**2+b_err**2)**0.5
		return x,x_err
def markline(ax,fontdict=None,loc='upper',c='gray',alpha=0.7,linewidth=1):
	linews=[3869,4102,4341,4640,4686,4861,5160,5304,5722,6088,6376,6563,7894]
	linenames=['NeIII',r'$\rm{H_\delta}$',r'$\rm{H_\gamma}$','NIII','HeII',r'$\rm{H_\beta}$','[FeVII]','[FeXIV]','[FeVII]','[FeVII]','FeX',r'$\rm{H_\alpha}$','[FeXI]']
	ylim=ax.get_ylim()
	ymin=ylim[0]+(ylim[1]-ylim[0])/20.
	ymax=ylim[1]-(ylim[1]-ylim[0])/20.
	print(ymin,ymax)
	j=0
	for linew,linename in zip(linews,linenames):
		print(ymin)
		ax.plot([linew]*2,[ymin,ymax],color=c,linewidth=linewidth,linestyle='dotted',alpha=alpha)
	#	ax.vline(linew,ymin,ymax,color=c,alpha=alpha,linewidth=linewidth,linestyle='dotted')	
		if loc=='upper':
			if j==0:
				ax.text(linew,ymax,linename,fontdict) #Is it ok when the parameter 'fontdict' equalling to None?
				j=1
			else:
				ax.text(linew,ymax*19/20,linename,fontdict)
				j=0
			continue
		if loc=='bottom':
			if j==0:
				ax.text(linew,ymax,linename,fontdict) #Is it ok when the parameter 'fontdict' equalling to None?
				j=1
			else:
				ax.text(linew,ymax*19/20,linename,fontdict)
				j=0
	return ':)'	
import multiprocessing
def multip(func,params,mode='apply'):
	if mode=='apply':
		pool=multiprocessing.Pool()
		pool.apply_async(func,args=params)
		pool.close()
		pool.join()
	return ':)'

def reddening_cal00(lam, ebv):
	#copy from the procedure pPXF
	"""
	Reddening curve of `Calzetti et al. (2000)
	<http://ui.adsabs.harvard.edu/abs/2000ApJ...533..682C>`_
	This is reliable between 0.12 and 2.2 micrometres.
	- LAMBDA is the restframe wavelength in Angstrom of each pixel in the
	  input galaxy spectrum (1 Angstrom = 1e-4 micrometres)
	- EBV is the assumed E(B-V) colour excess to redden the spectrum.
	In output the vector FRAC gives the fraction by which the flux at each
	wavelength has to be multiplied, to model the dust reddening effect.

	"""
	ilam = 1e4/lam  # Convert Angstrom to micrometres and take 1/lambda
	rv = 4.05  # C+00 equation (5)

	# C+00 equation (3) but extrapolate for lam > 2.2
	# C+00 equation (4) (into Horner form) but extrapolate for lam < 0.12
	k1 = rv + num.where(lam >= 6300, 2.76536*ilam - 4.93776,
						ilam*((0.029249*ilam - 0.526482)*ilam + 4.01243) - 5.7328)
	fact = 10**(-0.4*ebv*k1.clip(0))  # Calzetti+00 equation (2) with opposite sign

	return fact # The model spectrum has to be multiplied by this vector

def enob(value,N):
#return the value of in the format of N enob
	return ':)'
#	else:

#hdu=fits.open(r"D:\BaiduNetdiskDownload\variable\Transfer\0103\1115\SDSSJ1115_171124\J1115_blue.fits")
#wave=hdu[1].data['wavelength']
#flux=hdu[1].data['flux']
#plt.plot(wave,flux)
#flux_smooth=smooth(flux)
#plt.plot(wave,flux_smooth)
#print(len(wave),len(flux_smooth))
#plt.show() 

def download_sdss(path_out,mjd,plate,fiber,DR_N='dr15'):
	url1='https://data.sdss.org/sas/%s/sdss/spectro/redux/26/spectra/lite/%04d/spec-%04d-%d-%04d.fits'%(DR_N,plate,plate,mjd,fiber)
	url2='https://data.sdss.org/sas/%s/sdss/spectro/redux/103/spectra/lite/%04d/spec-%04d-%d-%04d.fits'%(DR_N,plate,plate,mjd,fiber)
	url3='https://data.sdss.org/sas/%s/sdss/spectro/redux/104/spectra/lite/%04d/spec-%04d-%d-%04d.fits'%(DR_N,plate,plate,mjd,fiber)
	url4='https://data.sdss.org/sas/%s/sdss/spectro/redux/v5_10_0/spectra/lite/%04d/spec-%04d-%d-%04d.fits'%(DR_N,plate,plate,mjd,fiber)
	url5='https://data.sdss.org/sas/%s/eboss/spectro/redux/v5_10_0/spectra/lite/%04d/spec-%04d-%d-%04d.fits'%(DR_N,plate,plate,mjd,fiber)
	
	status=0
	for url in [url1,url2,url3,url4,url5]:
		try:
			print(url)
			wget.download(url,out=path_out)
			status=1
			break
		except Exception as e:
			print(str(e))
			continue
	if status==0:
		print('Not found %s-%s-%s'%(plate,mjd,fiber))
	
	return status
def BHMgal_AGN(lgMgal, e_lgMgal): 
	#The relation between blackhole mass and galaxy total mass were taken from the equation (4) in RV2015 
	#NOTE the relation is suitable for AGN-hosts (derived from a sample of AGN), 
	#                              and may also be suitable for S/S0s galaxies with pseudobulges (see Figure 8 in the paper)
	#https://iopscience.iop.org/article/10.1088/0004-637X/813/2/82/pdf 
	lgM= 7.45+ 1.05*(lgMgal-11) 
	e_lgM= ( (1.05*e_lgMgal)**2+ 0.55**2)**0.5  #NOTE here we considering the intrinsic scattering using the RMS deviation from the paper 
	                                            # not the best-fit intrinsic scatter 
	return lgM, e_lgM  
def BHMgal_Bulge(lgMgal, e_lgMgal):
	#The relation between blackhole mass and galaxy total mass was taken from the euqation (6) in RV2015 
	#NOTE this relation is suitable for elliptical galaxies and spiral/S0 galaxy with calssical buges 
	#NOTE for these galaxies, there are other empirical relations between blackhole mass and bulge mass (such as kormendy and Ho 2013) 
	#NOTE for the elliptical galaxies, the bulge mass is the total stellar mass 
	#NOTE classical bulges: merge-built bulges? pseudo-bulges: made of material associated with the disk ? 
	lgM= 8.95+ 1.40*(lgMgal-11) 
	e_lgM= ( (1.40*e_lgMgal)**2+ 0.47**2)**0.5  #NOTE here we considering the inrinsic scattering using the best-fit inrinsic scatter 0.47 
	                                            # the RMS deviation is 0.6dex, it seems a few outliers enlarge the deviation? hence we use the 0.47 instead  
	return lgM, e_lgM 
 
def Msigma(sigma):
	#Kormendy & Ho 2013 Equation(7) 
	M=1e9*0.310*(sigma/200)**4.38
	M_up =1e9*0.347*(sigma/200)**4.67
	M_low=1e9*0.277*(sigma/200)**4.09
	return M,M_up,M_low
# M,M_up,M_low=Msigma(70) 
# print(M/1e6, M_up/1e6, M_low/1e6) 
def Msigma1(sigma):
	#McConnell & Ma 2013
	M=10**(8.32 + 5.64*num.log10(sigma/200))
	M_up=10**(8.32+0.05+5.96*num.log10(sigma/200))
	M_low=10**(8.32-0.05+5.32*num.log10(sigma/200))
	return M,M_up,M_low
# M,M_up,M_low=Msigma1(100) 
# print(M/1e6, M_up/1e6, M_low/1e6) 
def flux_rms(flux,ind=None):
	if not ind==None:
		flux=flux[ind]
	return (sum(flux**2)/len(flux))**0.5

def R_L(lam,L):
	K=1.527
	alpha=0.533
	R=10**(K+alpha*num.log10(lam*L/1e44))
	print('Units: lt days',R)
def FWHM(scales,sigmas,values):
	wave=num.arange(0,10000,0.1)
	flux=0
	if not scales.__class__ in [list,num.ndarray]:
		scales,sigmas,values=[scales],[sigmas],[values]

	for i in range(0,len(scales)):
		flux=flux+scales[i]*scipy.stats.norm.pdf(num.log(wave),num.log(values[i]),sigmas)

	ind1=(flux==num.max(flux))
	wavemax=wave[ind1]

	ind=flux>1/2.0*num.max(flux)	
	wave=wave[ind]


	FWHM=(wave[1]-wave[0])/wavemax*3e5

	return FWHM 


def upperXray(N, B, method='Bayesian', confidence_level=0.90): 
	#Based on the formula in P.Kraft 1991:  Determination of confidence limits  for experiments with low numbers or counts 

	#N is the observed photon counts in the source region 
	#B is the background in the source region , which can be estimated from other regions  

	#method, there are two methods 
	#         Bayesian, use the poisson distribution and Bayesian method  with a constant and nonnegative prior function 
	#         classical, use the poisson distribution and classical method 

	
	#NOTE the table1,2,3 in paper of Kraft1991, was Smin,Smax which make a minimum interval , here we just calculate the upperlimits 
	# by integrating from 0  

	if method=='Bayesian': 
		n=num.arange(N+1) 
		C= 1/sum( num.e**(-B) * B**n / special.factorial( n ,exact=True)) 
		def f(S):
			return C*num.e**(-(S+B))*(S+B)**N/ special.factorial(N,exact=True) 

		def I(x):
			return integrate.quad(f,0,x)[0]-confidence_level 
		
		upper=fsolve(I,N)[0] 

		print(upper, integrate.quad(f,0,upper))
		if integrate.quad(f,0,upper)[0]-confidence_level>0.0005: 
			raise Exception('ERRORs maybe occurred in the solve process ')

		return upper 

	if method=='classical': 
		def F(S):
			n=num.arange(N+1) 
			return sum( num.e**(-S) * S**n / special.factorial( n ,exact=True)) -(1-confidence_level)
		upper=fsolve(F,B)[0]   #NOTE how to define the defination region ?
	#	print(F(0),F(1),F(2),F(3),F(4),F(5),F(6),F(7))
		print(upper, 1-( F(upper)+1-confidence_level))    

		if  F(upper)>0.0005: 
			print(F(upper))
			raise Exception('ERRORs maybe occurred in the solve process ') 
	
		if upper >B : upper=upper-B 
		else:         upper=0 
		return upper 

#upperXray(9,5.5,method='Bayesian')
def CLXray(N, B, method='dichotomy', confidence_level=0.90,minimum=True):
	#calculate the minimum confidence limits at the given confidence level, according to Bayesian method of Kraft1991  
	N=float(N);B=float(B)
	
	if N==0:
		Smax=-num.log(1-confidence_level) #NOTE when N==0, the f(S)=e**-S
		print('results N=0',0,Smax)
		return 0,Smax  
	n=num.arange(N+1)
	C= 1/sum( num.e**(-B) * B**n / special.factorial( n ,exact=True)) 
	def f(S,Z):
		return C*num.e**(-(S+B))*(S+B)**N/ special.factorial(N,exact=True)-Z


	def I(x):
		return integrate.quad(f,0,x,args=0)[0]-confidence_level 

	Nini=1 if N-B<=0 else N-B
	Smax=fsolve(I,Nini)[0]  

	print(0,Smax, integrate.quad(f,0,Smax,args=0))
	if integrate.quad(f,0,Smax,args=0)[0]-confidence_level>0.0005: 
		raise Exception('ERRORs maybe occurred in the solve process ')

	if f(0,0)>=f(Smax,0):
		print('Results', 0,Smax, integrate.quad(f,0,Smax,args=0), f(0,0), f(Smax,0))
		return 0,Smax
	Smax_ini=Smax

	def solve_func(Smax): 
		#using the dichotomy to solve the f(Smin,0)=f(Smax,0), given a value of Smax
		if f(0,0)>=f(Smax,0):
			return 0 
		else:
			scale=0.01 
			x=num.arange(0,N-B,0.01) #NOTE the maximum of f(S,0) was reached when S=N-B, if N-B>0 or when S=0, if N-B<=0
			while 1: 
				ind=num.argmin(abs(f(x,0)-f(Smax,0)))
				if scale<1e-8: #NOTE maybe we can use the relative value of N-B?
					Smin=x[ind]
					break 
					
				if f(x[ind],0)-f(Smax,0)<=0 :
					x=num.arange(x[ind],x[ind+1],scale/100) 
				else: 
					x=num.arange(x[ind-1],x[ind],scale/100) 

				scale=scale/100
	#		print(Smin,Smax,f(Smin,0)-f(Smax,0))
			return  Smin
	
	if method=='dichotomy':
		def I(x):
			Smin=solve_func(x)
			return integrate.quad(f,Smin,x,args=0)[0]-confidence_level 
		Smax=fsolve(I,Smax_ini)[0] #here we can also use dichotomy,
		Smin=solve_func(Smax) 
	#	print('Results')
		print('Results',Smin,Smax,integrate.quad(f,Smin,Smax,args=0)[0],f(Smin,0),f(Smax,0))
		if integrate.quad(f,Smin,Smax,args=0)[0]-confidence_level>0.0005:
			raise Exception("There maybe error in the solving process") 
		if abs(f(Smax,0)-f(Smin,0))>0.0005: 
			raise Exception('ERRORs maybe occurred in the solving process of f(Smax,0)=f(Smin,0) ')
		return Smin,Smax

	elif method=='fsolve':  #NOTE To get proper results using the function fsolve, an proper initial-value is important,for example if the initial value is too large, the change between two iterations may be small 
		def I(x):
			Smin=fsolve(f,0,args=f(x,0))[0]
			return integrate.quad(f,Smin,x,args=0)[0]-confidence_level
		Smax=fsolve(I,Smax_ini)[0]
		Smin=fsolve(f,0,args=f(Smax,0)) 



		print('Results',Smin,Smax,integrate.quad(f,Smin,Smax,args=(0))[0],f(Smin,0),f(Smax,0))
		if integrate.quad(f,Smin,Smax,args=(0))[0]-confidence_level>0.0005: 
			raise Exception('ERRORs maybe occurred in the solve process ')
		if abs(f(Smax,0)-f(Smin,0))>0.0005: 
			raise Exception('ERRORs maybe occurred in the solving process of f(Smax,0)=f(Smin,0) ')
		return Smin,Smax 

#CLXray(6.0,3.0,method='dichotomy',confidence_level=0.90)
def SNxray(N, B, RN=20, Rin=60, Rout=200):
    #N, the total counts in the source region, i.e., the counts encircled by the circle with radius RN 
    #B, the background counts in the annular with Rin= Rin , Rout=Rout 

    #This was refer to the https://asc.harvard.edu/ciao/download/doc/detect_manual/cell_theory.html, but the symbol was different 
    frac= RN**2/(Rout**2-Rin**2) 
    sigN=1+(N+0.75)**0.5     # see the reference above and also Gehrels 1986 https://articles.adsabs.harvard.edu/pdf/1986ApJ...303..336G
    sigB=1+(B+0.75)**0.5
    SN= ( N - B*frac  ) / ( sigN**2+ sigB**2 * frac**2)**0.5  

    print(( N - B*frac  ) , ( sigN**2+ sigB**2 * frac**2)**0.5  )
    return SN 
