#some useful information for fitting emission lines

import numpy as num 

#[ArIII7135], 5876(HeI5876) ,[NII]5755,[OI]5577, HeII5411, HeII4685,[NeIII]3868,[NeIII]3970
#H-ε 3970, H-ζ 3889, 3835 
#NIII4100
#H-δ,4102
#N I, 5537.38
#Si I, 5539.1
# line at about 3425 ?
WAVE_LINE={'Hb':4862.683,'O3a':5008.240,'O3b':4960.295,\
		   'Ha':6564.61,'nii6549':6549.85,'nii6585':6585.28 ,'sii1':6718.29 ,'sii2':6732.67,\
		   'FeVII3759':3759,'FeVII5160':5160,'FeVII5722':5722,'FeVII6087':6087,'FeXI7894':7894,'FeXIV5304':5304,'FeV4071':4071,\
		   'FeX':6376,'OI6301':6301,'OI6364':6364,'HeII':4686.66,'NIII':4640}
#the first of the two in the key was tied to the second one
SCALE_TIED={'nii6585,nii6549':'2.96*p[%d]*'+str(6549.85/6585.28),'nii6549,nii6585':'1./2.96*p[%d]*'+str(6585.28/6549.85),\
			'O3a,O3b':'2.98*p[%d]*'+str(4960.295/5008.240),'O3b,O3a':'1.0/2.98*p[%d]*'+str(5008.240/4960.295),\
			'sii1,sii2':'p[%d]*'+str(6732.67/6718.29),'sii2,sii1':'p[%d]*'+str(6718.29/6732.67),\
			'OI6301,OI6364':'3*p[%d]*'+str(6364./6301.),'OI6364,OI6301':'1.0/3*p[%d]*'+str(6301./6364.)}

#NOTE the data below was taken from NIST 
#The deifinition of Aki ? 
#NII6583和6548同一上能级，Aki 分别为2.91e-3,9.84e-4  #NOTE NOTE 29.1/9.84*6548/6583~2.94? 和下面的dict里的值不一致？ 这里其实没问题，lineratio~Aki1/Aki2*lam2/lam1 (跃迁系数之比乘以每个光子的能量之比)
                                                   #但是NIST上并不是最新的参考文献的结果，现在常用的是2.98 https://ui.adsabs.harvard.edu/abs/2007MNRAS.374.1181D/abstract 
#O3a    和O3b 同一上能级，Aki 分别为1.81e-2,6.21e-3, #NOTE NOTE 和上面[NII]的问题类似
#OIII4363的下能级为O3a
#OI6301和6364 同一上能级，Aki分别为5.63e-3, 1.82e-3 
#NOTE the wavelength below from NIST, some are observed wavelength while some others are theoritical ones. we didn't make this consistent for all 
lam_FeVII=[6087,5720.7,3758.92,3586.3]
lam_FeV  =[4071.29]
lam_FeVI =[5176.04,5145.75,3889.4,3813.54,3662.5]#4972.47,4967.14在o3附近 #5176 和4967 同一上能级且发射系数一个是6.2e-1，2.5e-1. 5145和4972同一上能级且发射系数为2.6e-1和2.4e-1. 3889和3913同一上能级，发射系数分别为5.8e-1.3.6e-1
lam_FeX  =[6374.5]
lam_NeV  =[3345.83,3425.87] 
lam_NeII  =[3868.76,3967.47]
lam_FeXI =[7891.8]

WAVE_LINE={'Hb':4861.3,'O3a':5006.843,'O3b':4958.911,'OIII4363':4363.209,'Ha':6562.7,'nii6549':6548.05,'nii6585':6583.45,'sii1':6716.440,'sii2':6730.815,'OI6301':6300.304,'OI6364':6363.776,'HeII':4685.7,'NIII':4640.64,
           'Hg':4340.472,'Hd':4101.7, 'H5':3970.075,'H6':3889.025,'H7':3835.397,'H8':3797.9,'H9':3770.6,'H10':3750.1}

#the first of the two in the key was tied to the second one
SCALE_TIED={'nii6585,nii6549':'2.94408*p[%d]','nii6549,nii6585':'1.0/2.94408*p[%d]',\
			'O3a,O3b'        :'2.95147*p[%d]','O3b,O3a'        :'1.0/2.95147*p[%d]',\
			'sii1,sii2'      : 'p[%d]*'+str(6732.67/6718.29),'sii2,sii1':'p[%d]*'+str(6718.29/6732.67),\
			'OI6301,OI6364'  :'3.1560*p[%d]','OI6364,OI6301'  :'1.0/3.1560*p[%d]'} #NOTE SII shouldn't be tied toegther since they have different upper energy level 
#the templates used to tied the second emission lines, and I think we can simply ignore the difference between the scale_ratio and 
#the flux_ratio
#NOTE, the flux ratio of [OIII] doublets  and [NII] doublets are fixed at 2.98 and 2.96 based on the literature( 
#       https://doi.org/10.3847/1538-4365/ab298b
#       https://ui.adsabs.harvard.edu/abs/2000MNRAS.312..813S/abstract) 

def composite_spectra_width(linename):
#the follow data was collected from the https://iopscience.iop.org/article/10.1086/321167/pdf
# the following sigma may be wrong and should be gotten by fitting the composite spectra not just using the sigma_lamba given in the paper 
	if linename=='Ha': return 47.39/6564.93/2.355,6564.93# whether the sigma here is needed to divided by the light velocity or not?
#	if linename=='O3a': return 6.04/5008.22/2.355,5008.22
	if linename=='O3a': return 0.0005540766120768197,5008.22 #sigma is the fitting results of composite spectra, lambda was taken from that paper directly
	if linename=='O3b': return 0.0005540766120768197,4960.36 #sigma is the fitting results of compoiste spectra 
	if linename=='Hb': return 40.44/4853.13/2.355,4853.13
	if linename=='FeVII3759': return 3.71/3758.46/2.355,3758.46
	if linename=='FeVII5160': return 3.95/5160.81/2.355,5160.81
	if linename=='FeVII5722': return 5.19/5723.74/2.355,5723.74
	if linename=='FeVII6087': return 3.78/6087/2.355,6087
	if linename=='FeX': return 4/6376.0/2.355,6376 #just assume it's 4, not the results from that paper
	if linename=='FeXIV5304': return 5.34/5313.82/2.355,5313.82
	if linename=='FeV4071': return 3.20/4070.71/2.355,4070.71
	if linename=='HeII': return 5.92/4686.66/2.355,4686.66
	if linename=='FeXI7894': return 4/7894.0/2.355,7894 #just assume it's 4, not the results from that paper
#	if linename=='NIII':

def linefit_range(linename):
	if linename=='Ha': return [6400,6760]#[6150, 6950]#[6400,6760] #[6250, 6850] #[6320,6850]#[6400,6760] #[6250,6850] # [6390,6770] #[6250,6800]#[6089,7000] #[6390,6770]#[6089,7000]#  # #
	if linename=='Hb': return [4750,5070]#[4715,5100]# [4750,5070]#  # [4500,5150] # #[4500,5150] #[4550,5250] #[4600,5200] #[4750,5070]
	if linename=='FeVII3759': return [3735,3795]
	if linename=='FeV4071': return [4060,4083]
	if linename=='FeVII5160': return [5130,5190]
	if linename=='FeXIV5304': return [5258,5360]#[5265,5340]
	if linename=='FeVII5722': return [5690,5750]
	if linename=='FeVII6087': return [6040,6140]#[6050,6130]
	if linename=='FeX': return [6280,6450]
	if linename=='FeXI7894': return [7840,7940]
	if linename=='HeII': return [4646,4770]
	if linename=='NIII': return [4530,4680]
	if linename=='HeII+NIII': return [4530,4770]
	if linename=='o3': return [4940,5035]
	if linename=='sii': return [6690,6752]
	if linename=='FeX3455': return [3405,3521]
	if linename=='FeXI3484': return [3405,3521]
def linefit_cut(wave,flux,err,linename,cutregion=None):
	if cutregion  is None:
		if '+' not in linename:
			region=linefit_range(linename)
			ind=(wave>region[0]) & (wave<region[1])
		else:
			lnames=linename.split('+')
			ind=wave<0
			for lname in lnames:
				region=linefit_range(lname)
				ind=ind | ( (wave>region[0]) & (wave<region[1]) ) 
	else: 
		ind=wave<0
		for region in cutregion: 
			ind=ind | ( (wave>region[0]) & (wave<region[1])  )
	return wave[ind],flux[ind],err[ind]

def line_from_fitrange(wave):
	if (wave[0]<6563) & (wave[1]>6563):
		return 'halpha'
	if (wave[0]<4861) & (wave[1]>4861):
		return 'hbeta'
	if (wave[0]<6087) & (wave[1]>6087):
		return 'FeVII6087'
	if (wave[0]<6376) & (wave[1]>6376):
		return 'FeX'
	if (wave[0]<4686) & (wave[1]>4686):
		return 'HeII'
	if (wave[0]<4640) & (wave[1]>4640):
		return 'NIII'
	if (wave[0]<3759) & (wave[1]>3759):
		return 'FeVII3759'
	if (wave[0]<5160) & (wave[1]>5160):
		return 'FeVII5160'
	if (wave[0]<5304) & (wave[1]>5304):
		return 'FeXIV5304'
	if (wave[0]<5722) & (wave[1]>5722):
		return 'FeVII5722'
	if (wave[0]<4071) & (wave[1]>4071):
		return 'FeV4071'
	if (wave[0]<7894) & (wave[1]>7894):
		return 'FeXI7894'
def nearby_conti(wave,linename):
	if linename=='Ha': return [6120,6250]
	if linename=='Hb': return [5100,5200]
	if linename=='FeVII3759': return [3735,3795]
	if linename=='FeV4071': return [4050,4083]
	if linename=='FeVII5160': return [5115,5205]
	if linename=='FeXIV5304': return [5265,5355]
	if linename=='FeVII5722': return [5675,5765]
	if linename=='FeVII6087': return [6050,6130]
	if linename=='FeX': return [6230,6450]
	if linename=='FeXI7894': return [7850,7930]
	if linename=='HeII': return [4646,4730]
	if linename=='NIII': return [4600,4680]
	if linename=='o3': return [4945,5030]
	if linename=='sii': return [6690,6752]