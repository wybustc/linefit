from astropy.io import fits
import numpy as num
import os,math
from PyAstronomy import pyasl
from sfdmap import ebv as ebv_value
import matplotlib.pyplot as plt
def create_fits(datas,params,path_savefile):
	# create the primary HDU
	prihdr=fits.Header()
	if params[0]==None:
		prihdu=fits.PrimaryHDU(data=datas[0])
	else:
		for key in params[0].keys():
			prihdr[key]=params[0][key]
		prihdu=fits.PrimaryHDU(data=datas[0],header=prihdr)

#   create other hdus....
	tbhdus=[]
	for i in range(1,len(datas)):
		if datas[i]==None:
			raise Exception('when not primary HDU, datas[i] should not be None')
		datacols=[]
		for key in datas[i].keys():
			if 'disp' in datas[i][key].keys():
				datacol=fits.Column(name=key,format=datas[i][key]['fmt'],unit=datas[i][key]['unit'],array=datas[i][key]['data'],disp=datas[i][key]['disp'])
			else:
				datacol=fits.Column(name=key,format=datas[i][key]['fmt'],unit=datas[i][key]['unit'],array=datas[i][key]['data'])
			datacols.append(datacol)
		cols=fits.ColDefs(datacols)
		tbhdu=fits.BinTableHDU.from_columns(cols)

		if not params[i]==None:
			for key in params[i].keys():
				tbhdu.header[key]=params[i][key]

		tbhdus.append(tbhdu)

	tbhdulist=fits.HDUList([prihdu]+tbhdus)
	if os.path.exists(path_savefile):
		os.remove(path_savefile)
	tbhdulist.writeto(path_savefile)
	return ':)'

def covert1241():
	#covert the format of J1241_150417 (from Marie Lau) to our format
	hdu=fits.open(r"D:\new_infrared\orginaldata\ALL\SDSSJ124134.26+442639.2\J1241_F.fits")
	wave=hdu[2].data
	flux=hdu[0].data
	err=hdu[1].data
	params={0:{},1:{'wave_point':0,'wave_connect':0,'ratio':1,'ratio_err':0,'merge_status':'good'}}#we don't know the detail of merge, but it won't influence much,so I just set this value at will
	datas={0:None,1:{'wave_dbsp':{'data':wave,'fmt':'D','unit':'Angstrom'},\
					'flux_dbsp':{'data':flux*1e-17,'fmt':'D','unit':'erg/s/cm^2/A'},\
					'err_dbsp':{'data':err*1e-17,'fmt':'D','unit':'erg/s/cm^2/A'}}}
	path=r"D:\new_infrared\orginaldata\ALL\SDSSJ124134.26+442639.2\J1241.fits"
	create_fits(datas,params,path)	
def fromTing2ppxf(path,path2,path_sdss,source='Ning'):
	print('Note: please make sure that whether we need to do the redshift-correction when do the format change!!!')
	a=input('Press any to continue')
	hdu=fits.open(path_sdss)
	redshift=hdu[2].data['Z'][0]
	w_sdss=10**hdu[1].data['loglam']/(1+redshift)
	f_sdss=hdu[1].data['flux']*(1+redshift)
	e_sdss=hdu[1].data['ivar']**-0.5*(1+redshift)
	ra,dec=hdu[0].header['plug_ra'],hdu[0].header['plug_dec']
	mw_ebv=ebv_value((ra,dec),unit='degree')
	f_sdss=pyasl.unred(w_sdss*(1+redshift),f_sdss,mw_ebv)
	e_sdss=pyasl.unred(w_sdss*(1+redshift),e_sdss,mw_ebv)

	if source=='Ning': pass
	elif source=='twfit': redshift=0 #In the python code of twfit, I have include the redshift correction for flux
	else: raise Exception('No such mode')
	hdu=fits.open(path)
	wave=num.e**hdu[1].data['wave'][0]
	flux=hdu[1].data['flux'][0]*(1+redshift)
	err=1/hdu[1].data['ivar'][0]**0.5*(1+redshift)
	stellar=hdu[1].data['starlight'][0]*(1+redshift)
	powlaw=hdu[1].data['POWLAW'][0]*(1+redshift)
#	stellar=hdu[1].data['starlight'][0]
	spec=hdu[1].data['spec'][0]*(1+redshift)
	err_stellar=num.zeros_like(stellar)
	
#	ind1=(wave>3980)&(wave<3985)
#	ind2=(wave>6834)&(wave<6838)
#	ind3=(wave>6957.5)&(wave<6965)
#	ind4=(wave>8092)&(wave<8098)

#	ind=ind1 | ind2| ind3| ind4
#	wave,flux,err,stellar,powlaw,spec,err_stellar=wave[ind==False],flux[ind==False],err[ind==False],stellar[ind==False],powlaw[ind==False],spec[ind==False],err_stellar[ind==False]

	plt.plot(wave)
	plt.plot(w_sdss)
	plt.show()
	plt.plot(wave,flux,'r',w_sdss,f_sdss,'b')
	plt.show()
	plt.plot(wave,err,'r',w_sdss,e_sdss,'b')
	plt.show()
	plt.plot(wave,flux,wave,stellar+powlaw)
	plt.show()
	plt.plot(wave,stellar+powlaw,wave,spec)
	plt.show()


	params={0:{'redshift':redshift,'ebv':mw_ebv},1:{'stellar_err':True,'chi/dof':sum( ( (flux-spec)/err )**2 )/(len(flux)-13 )}}
	datas={0:None,1:{'wave':{'data':wave,'fmt':'D','unit':'Angstrom'},\
					'original_data':{'data':flux,'fmt':'D','unit':'1e-17erg/s/cm^2/A'},\
					'err_sdss':{'data':err,'fmt':'D','unit':'1e-17erg/s/cm^2/A'},\
					'stellar':{'data':stellar,'fmt':'D','unit':'1e-17erg/s/cm^2/A'},\
					'err_stellar':{'data':err_stellar,'fmt':'D','unit':'1e-17erg/s/cm^2/A'},\
					'powlaw':{'data':powlaw,'fmt':'D','unit':'1e-17erg/s/cm^2/A'},\
					'spec_fit':{'data':spec,'fmt':'D','unit':'1e-17erg/s/cm^2/A'}}}
	create_fits(datas,params,path2)
	return ':)'

	
#fromPypeit2old(path,mode='fluxc',slitname='SPAT'):
#	hdu=fits.open(path)
#	if mode=='fluxc':
#		params={0:{'DATE':hdu[0].header['DATE']},1:None}
#		datas={0:None,1:}
