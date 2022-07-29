#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U bmaclient')


# In[2]:


import os
import json

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from bmaclient import MonitoringAPI
from bmaclient.utils import get_api_key
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

import datetime as dt
import scipy
from scipy import signal
import statsmodels.api as sm


# In[3]:


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# client = MonitoringAPI(api_key='1JPZxKW5.RCpwvwK5O4T5hYFTzfPpSp9o2PdTtRwa')



def gps_data(start, end, station):
    client = MonitoringAPI(api_key='1JPZxKW5.RCpwvwK5O4T5hYFTzfPpSp9o2PdTtRwa')
    results = client.fetch_gps_position(
    timestamp__gte=start,
    timestamp__lt=end,
    station =station,
    nolimit=True,
    aggregation='mean')
    return results


# In[4]:


# Linear Regression calculation

def lin_reg(station):
    #Change timestamp data type to numerical value
    station['timestamp'] = pd.to_datetime(station['timestamp'])
    station['timestamp']=station['timestamp'].map(dt.datetime.toordinal)
    #Median filter
    station['east']= scipy.signal.medfilt(station['east'],kernel_size=3)
    station['north']= scipy.signal.medfilt(station['north'],kernel_size=3)

    #Linear Regression
    X = sm.add_constant(station['timestamp'])
    modelgx = sm.OLS(station['east'],X)
    resultsgx = modelgx.fit()
    y_predictgx = resultsgx.params[0] + resultsgx.params[1]*station['east']

    X = sm.add_constant(station['timestamp'])
    modelgy = sm.OLS(station['north'],X)
    resultsgy = modelgy.fit()
    y_predictgy = resultsgy.params[0] + resultsgy.params[1]*station['timestamp']
    
    return resultsgx.params[0],resultsgx.params[1],resultsgy.params[0],resultsgy.params[1]


# In[5]:


def lin_regEW(station):
    #Change timestamp data type to numerical value
    station['timestamp'] = pd.to_datetime(station['timestamp'])
    station['timestamp']=station['timestamp'].map(dt.datetime.toordinal)
    #Median filter
    station['east']= scipy.signal.medfilt(station['east'],kernel_size=3)
    station['up']= scipy.signal.medfilt(station['up'],kernel_size=3)

    #Linear Regression
    X = sm.add_constant(station['timestamp'])
    modelgx = sm.OLS(station['east'],X)
    resultsgx = modelgx.fit()
    y_predictgx = resultsgx.params[0] + resultsgx.params[1]*station['east']

    X = sm.add_constant(station['timestamp'])
    modelgy = sm.OLS(station['up'],X)
    resultsgy = modelgy.fit()
    y_predictgy = resultsgy.params[0] + resultsgy.params[1]*station['timestamp']
    
    return resultsgx.params[0],resultsgx.params[1],resultsgy.params[0],resultsgy.params[1]


# In[6]:


def lin_regNS(station):
    #Change timestamp data type to numerical value
    station['timestamp'] = pd.to_datetime(station['timestamp'])
    station['timestamp']=station['timestamp'].map(dt.datetime.toordinal)
    #Median filter
    station['north']= scipy.signal.medfilt(station['north'],kernel_size=3)
    station['up']= scipy.signal.medfilt(station['up'],kernel_size=3)

    #Linear Regression
    X = sm.add_constant(station['timestamp'])
    modelgx = sm.OLS(station['north'],X)
    resultsgx = modelgx.fit()
    y_predictgx = resultsgx.params[0] + resultsgx.params[1]*station['north']

    X = sm.add_constant(station['timestamp'])
    modelgy = sm.OLS(station['up'],X)
    resultsgy = modelgy.fit()
    y_predictgy = resultsgy.params[0] + resultsgy.params[1]*station['timestamp']
    
    return resultsgx.params[0],resultsgx.params[1],resultsgy.params[0],resultsgy.params[1]


# In[ ]:





# In[7]:


# Vector plot

def vector_res(x,y):
    # Select length of axes and the space between tick labels
    xmin, xmax, ymin, ymax = -0.0025, 0.0025, -0.0009, 0.0009
    ticks_frequency = 0.001
    origin = (0,0)

    # Plot points
    fig, ax = plt.subplots(figsize=(10, 10))


    fig.add_subplot(ax)
    ax.quiver(*origin, x, y, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax.set_xlim(-0.001,0.001)
    ax.set_ylim(-0.001,0.001)

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')


    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    # Create 'x' and 'y' labels placed at the end of the axes
    ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
    ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)
    ax.legend()

    # Draw major and minor grid lines
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

    # Draw arrows
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
    ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)

    plt.show()


# In[8]:


def read_grd(filename):
    with open(filename) as infile:
        name = infile.readline()

        nx = infile.readline().split()
        ncols = int(nx[0])
        nrows = int(nx[1])

        xlo = infile.readline().split()
        xmin = float(xlo[0])
        xmax = float(xlo[1])

        ylo = infile.readline().split()
        ymin = float(ylo[0])
        ymax = float(ylo[1])

        zlo = infile.readline().split()
        zmin = float(zlo[0])
        zmax = float(zlo[1])

        result = []
        for i in range(nrows):
            arr = []
            while True:
                string = infile.readline().split()
                if string:
                    arr += [float(s) for s in string]
                else:
                    break
            result.append(arr)

    longitude = np.linspace(xmin, xmax, ncols)
    latitude = np.linspace(ymin, ymax, nrows)

    return longitude, latitude, np.array(result)


def main():
    filename = 'out.grd'
    val = read_grd(filename)


    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(val[0], val[1])
    Z = val[2]
    positions = np.vstack(map(np.ravel, [X, Y, Z])).T
    print(positions.shape)

    res = []
    for index, item in enumerate(positions.tolist()):
        if index % 2 != 0:
            res.append(item)

    print(len(res))

    with open('data.json', 'w') as f:
        json.dump(res, f, indent=4)
    surf = ax.plot_surface(X, Y, val[2], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    

    plt.show()

if __name__ == '__main__':
    main()


# In[9]:


#Running the function for each location

#Deles No Data
#deles = gps_data('2022-03-01','2022-03-10','deles')
#deles = pd.DataFrame(deles)
#vecdel = lin_reg(deles)

grawah = gps_data('2022-03-01','2022-04-10','grawah')
grawah = pd.DataFrame(grawah)
vecgra = lin_reg(grawah)
grawah = gps_data('2022-03-01','2022-04-10','grawah')
grawah = pd.DataFrame(grawah)
vecgraew = lin_regEW(grawah)
grawah = gps_data('2022-03-01','2022-04-10','grawah')
grawah = pd.DataFrame(grawah)
vecgrans = lin_regNS(grawah)

selo = gps_data('2022-03-01','2022-04-10','selo')
selo = pd.DataFrame(selo)
vecsel = lin_reg(selo)
selo = gps_data('2022-03-01','2022-04-10','selo')
selo = pd.DataFrame(selo)
vecselew = lin_regEW(selo)
selo = gps_data('2022-03-01','2022-04-10','selo')
selo = pd.DataFrame(selo)
vecselns = lin_regEW(selo)

klatakan = gps_data('2022-03-01','2022-04-10','klatakan')
klatakan = pd.DataFrame(klatakan)
veckla = lin_reg(klatakan)
klatakan = gps_data('2022-03-01','2022-04-10','klatakan')
klatakan = pd.DataFrame(klatakan)
vecklaew = lin_regEW(klatakan)
klatakan = gps_data('2022-03-01','2022-04-10','klatakan')
klatakan = pd.DataFrame(klatakan)
vecklans = lin_regEW(klatakan)

labuhan = gps_data('2022-03-01','2022-04-10','labuhan')
labuhan = pd.DataFrame(labuhan)
veclab = lin_reg(labuhan)
labuhan = gps_data('2022-03-01','2022-04-10','labuhan')
labuhan = pd.DataFrame(labuhan)
veclabew = lin_regEW(labuhan)
labuhan = gps_data('2022-03-01','2022-04-10','labuhan')
labuhan = pd.DataFrame(labuhan)
veclabns = lin_regEW(labuhan)

# Kendit No Data
#kendit = gps_data('2022-03-01','2022-05-10','kendit')
#kendit = pd.DataFrame(kendit)
#vecken = lin_reg(kendit)

babadan = gps_data('2022-03-01','2022-04-10','babadan')
babadan = pd.DataFrame(babadan)
vecbab = lin_reg(babadan)
babadan = gps_data('2022-03-01','2022-04-10','babadan')
babadan = pd.DataFrame(babadan)
vecbabew = lin_regEW(babadan)
babadan = gps_data('2022-03-01','2022-04-10','babadan')
babadan = pd.DataFrame(babadan)
vecbabns = lin_regEW(babadan)

pasbub = gps_data('2022-03-01','2022-04-10','pasarbubar')
pasbub = pd.DataFrame(pasbub)
vecpas = lin_reg(pasbub)
pasbub = gps_data('2022-03-01','2022-04-10','pasarbubar')
pasbub = pd.DataFrame(pasbub)
vecpasew = lin_regEW(pasbub)
pasbub = gps_data('2022-03-01','2022-04-10','pasarbubar')
pasbub = pd.DataFrame(pasbub)
vecpasns = lin_regNS(pasbub)

jrakah = gps_data('2022-03-01','2022-04-10','jrakah')
jrakah = pd.DataFrame(jrakah)
vecjra = lin_reg(jrakah)
jrakah = gps_data('2022-03-01','2022-04-10','jrakah')
jrakah = pd.DataFrame(jrakah)
vecjraew = lin_regEW(jrakah)
jrakah = gps_data('2022-03-01','2022-04-10','jrakah')
jrakah = pd.DataFrame(jrakah)
vecjrans = lin_regEW(jrakah)

plawangan = gps_data('2022-03-01','2022-04-10','plawangan')
plawangan = pd.DataFrame(plawangan)
vecpla = lin_reg(plawangan)
plawangan = gps_data('2022-03-01','2022-04-10','plawangan')
plawangan = pd.DataFrame(plawangan)
vecplaew = lin_regEW(plawangan)
plawangan = gps_data('2022-03-01','2022-04-10','plawangan')
plawangan = pd.DataFrame(plawangan)
vecplans = lin_regNS(plawangan)


# In[13]:


def main():
    filename = 'out.grd'
    val = read_grd(filename)



    X, Y = np.meshgrid(val[0], val[1])
    Z = val[2]
    positions = np.vstack(map(np.ravel, [X, Y, Z])).T
    print(positions.shape)

    res = []
    for index, item in enumerate(positions.tolist()):
        if index % 2 != 0:
            res.append(item)

    print(len(res))

    with open('data.json', 'w') as f:
        json.dump(res, f, indent=4)
        
   
    from pyproj import Proj
    e = np.array([440572.96, 439486.48, 439311.11, 440947.98, 437373.01, 438783.15, 438800.79, 434974.70, 439212.52, 436184.10, 437289.14])
    n = np.array([9164285.46, 9168551.49, 9167633.67, 9163442.39, 9167365.86, 9164632.56, 9165508.17, 9168037.08, 9167069.72, 9171233.72, 9161462.13])
    s = ['  Deles0', '  Grawah', '  Selokopo', '  Deles1', '  Klatakan', '  Labuhan', '  Kendit', '  Babadan', '  Pasar Bubar', '  Jrakah', '  Plawangan']
    
    #Vector parameter
    
    #orgdel = np.array([e[0],n[0]])
    #delx = vecdel[0]-orgdel[0]
    #dely = vecdel[2]-orgdel[1]

    orggra = np.array([e[1],n[1]])
    grax = vecgra[1]*orggra[0]
    gray = vecgra[3]*orggra[1]
    
    orgsel = np.array([e[2],n[2]])
    selx = vecsel[1]*orgsel[0]
    sely = vecsel[3]*orgsel[1]
    
    orgkla = np.array([e[4],n[4]])
    klax = veckla[1]*orgkla[0]
    klay = veckla[3]*orgkla[1]
    
    orglab = np.array([e[5],n[5]])
    labx = veclab[1]*orglab[0]
    laby = veclab[3]*orglab[1]
    
    #orgken = np.array([e[6],n[6]])
    #kenx = vecken[0]-orgken[0]
    #keny = vecken[2]-orgken[1]
    
    orgbab = np.array([e[7],n[7]])
    babx = vecbab[1]*orgbab[0]
    baby = vecbab[3]*orgbab[1]
    
    orgpas = np.array([e[8],n[8]])
    pasx = vecpas[1]*orgpas[0]
    pasy = vecpas[3]*orgpas[1]
    
    orgjra = np.array([e[9],n[9]])
    jrax = vecjra[1]*orgjra[0]
    jray = vecjra[3]*orgjra[1]
    
    orgpla = np.array([e[10],n[10]])
    plax = vecpla[1]*orgpla[0]
    play = vecpla[3]*orgpla[1]


    fig, ax = plt.subplots(figsize=(10,10))
    for i, txt in enumerate(s):
        ax.scatter(e[i],n[i], color='crimson', s=12, marker='.')
        ax.annotate(txt, (e[i],n[i]), size=7, color="#181818")   
    fig.add_subplot(ax)
    contours = ax.contour(X, Y, Z, colors='peru',antialiased='False', levels=15, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8)
    ax.set_title('GPS Vector')
    
    #Vector Plot
    ax.quiver(*orggra, grax, gray, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax.quiver(*orgsel, selx, sely, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax.quiver(*orgkla, klax, klay, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax.quiver(*orglab, labx, laby, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax.quiver(*orgbab, babx, baby, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax.quiver(*orgpas, pasx, pasy, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax.quiver(*orgjra, jrax, jray, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax.quiver(*orgpla, plax, play, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    
    plt.savefig('GPS Vector.jpg')
if __name__ == '__main__':
    main()


# In[11]:


#cross section plot

def main():
    filename = 'out.grd'
    val = read_grd(filename)
    X, Y = np.meshgrid(val[0], val[1])
    Z = val[2]
    positions = np.vstack(map(np.ravel, [X, Y, Z])).T
    scale = 1
   
    e = np.array([440572.96, 439486.48, 439311.11, 440947.98, 437373.01, 438783.15, 438800.79, 434974.70, 439212.52, 436184.10, 437289.14])
    n = np.array([9164285.46, 9168551.49, 9167633.67, 9163442.39, 9167365.86, 9164632.56, 9165508.17, 9168037.08, 9167069.72, 9171233.72, 9161462.13])
    u = np.array([1519, 2045, 2363, 1399, 1924, 1730, 2308, 1281.8, 2679, 1281, 1165])
    s = ['  Deles0', '  Grawah', '  Selokopo', '  Deles1', '  Klatakan', '  Labuhan', '  Kendit', '  Babadan', '  Pasar Bubar', '  Jrakah', '  Plawangan']
    
    #orgdel = np.array([e[0],n[0]])
    #delx = vecdel[0]-orgdel[0]
    #dely = vecdel[2]-orgdel[1]
    
    orggrae = np.array([e[1],u[1]])
    orggran = np.array([n[1],u[1]])
    graxe = vecgraew[1]*orggrae[0]
    graye = vecgraew[3]*orggrae[1]
    graxn = vecgrans[1]*orggran[0]
    grayn = vecgrans[3]*orggran[1]
    
    orgsele = np.array([e[2],u[2]])
    orgseln = np.array([n[1],u[1]])
    selxe = vecselew[1]*orgsele[0]
    selye = vecselew[3]*orgsele[1]
    selxn = vecselns[1]*orgseln[0]
    selyn = vecselns[3]*orgseln[1]
    
    orgklae = np.array([e[4],u[4]])
    orgklan = np.array([n[4],u[4]])
    klaxe = vecklaew[1]*orgklae[0]
    klaye = vecklaew[3]*orgklae[1]
    klaxn = vecklans[1]*orgklan[0]
    klayn = vecklans[3]*orgklan[1]
    
    orglabe = np.array([e[5],u[5]])
    orglabn = np.array([n[5],u[5]])
    labxe = veclabew[1]*orglabe[0]
    labye = veclabew[3]*orglabe[1]
    labxn = veclabns[1]*orglabn[0]
    labyn = veclabns[3]*orglabn[1]
    
    #orgken = np.array([e[6],u[6]])
    #kenx = veckenew[0]-orgken[0]
    #keny = veckenew[2]-orgken[1]
    
    orgbabe = np.array([e[7],u[7]])
    orgbabn = np.array([n[7],u[7]])
    babxe = vecbabew[1]*orgbabe[0]
    babye = vecbabew[3]*orgbabe[1]
    babxn = vecbabns[1]*orgbabn[0]
    babyn = vecbabns[3]*orgbabn[1]

    
    orgpase = np.array([e[8],u[8]])
    orgpasn = np.array([n[8],u[8]])
    pasxe = vecpasew[1]*orgpase[0]
    pasye = vecpasew[3]*orgpase[1]
    pasxn = vecpasns[1]*orgpasn[0]
    pasyn = vecpasns[3]*orgpasn[1]
    
    orgjrae = np.array([e[9],u[9]])
    orgjran = np.array([n[9],u[9]])
    jraxe = vecjraew[1]*orgjrae[0]
    jraye = vecjraew[3]*orgjrae[1]
    jraxn = vecjrans[1]*orgjran[0]
    jrayn = vecjrans[3]*orgjran[1]
    
    orgplae = np.array([e[10],u[10]])
    orgplan = np.array([n[10],u[10]])
    plaxe = vecplaew[1]*orgplae[0]
    playe = vecplaew[3]*orgplae[1]
    plaxn = vecplans[1]*orgplan[0]
    playn = vecplans[3]*orgplan[1]
    
    fig, ax = plt.subplots(2, figsize=(10,10))
    ax[0].plot(X[0,:],Z[50,:])
    ax[0].set_title('East-West Cross Section')
    for i, txt in enumerate(s):
        ax[0].scatter(e[i],u[i], color='crimson', s=12, marker='.')
        ax[0].annotate(txt, (e[i],u[i]), size=7, color="#181818") 
    ax[0].quiver(*orggrae, graxe, graye, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax[0].quiver(*orgsele, selxe, selye, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax[0].quiver(*orgklae, klaxe, klaye, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax[0].quiver(*orglabe, labxe, labye, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax[0].quiver(*orgbabe, babxe, babye, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax[0].quiver(*orgpase, pasxe, pasye, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax[0].quiver(*orgjrae, jraxe, jraye, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax[0].quiver(*orgplae, plaxe, playe, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    
    
    ax[1].plot(Y[:,0],Z[:,50])
    ax[1].set_title('North-South Cross Section')
    for i, txt in enumerate(s):
        ax[1].scatter(n[i],u[i], color='crimson', s=12, marker='.')
        ax[1].annotate(txt, (n[i],u[i]), size=7, color="#181818") 
    ax[1].quiver(*orggran, graxn, grayn, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax[1].quiver(*orgseln, selxn, selyn, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax[1].quiver(*orgklan, klaxn, klayn, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax[1].quiver(*orglabn, labxn, labyn, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax[1].quiver(*orgbabn, babxn, babyn, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax[1].quiver(*orgpasn, pasxn, pasyn, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax[1].quiver(*orgjran, jraxn, jrayn, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    ax[1].quiver(*orgplan, plaxn, playn, angles='xy', scale_units='xy',scale = 1, width=0.003, color='#212121')
    
    
    fig.tight_layout()
    plt.savefig('Cross Section.jpg')
if __name__ == '__main__':
    main()


# In[ ]:




