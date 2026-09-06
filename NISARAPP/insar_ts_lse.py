#!/usr/bin/env python3

#Cunren Liang, JPL/Caltech, 10-OCT-2017
#A program for least squares estimation of the InSAR time series.
#Gaussian filter is exactly normpdf(X,MU,SIGMA) in matlab, where SIGMA is filt and MU=0.
#So can evaluate this filter by looking at plotting normpdf in matlab.
#e.g.
#x=-3:0.01:3;
#filt=0.75;
#y=normpdf(x,0,filt);
#plot(x,y);


#example command:
#insar_ts_lse.py -plist ../plist.txt -unw "data/*-*.unw.geo" -output velocity.unw.geo -wvl 0.05546576 -sigma 0.8

#plist.txt contains the list of used pairs and its format is mdate-sdate. a line is considered as valid if it contains -
# 150219-150402
# 150219-150514
# 150219-151015
# 150219-151126
# 150402-150514
# 150402-151015
# 150402-151126


import os
import sys
import glob
import copy
import ntpath
import shutil
import pickle
import datetime
import argparse
import numpy as np
import numpy.matlib
from xml.etree.ElementTree import ElementTree


def runCmd(cmd, silent=0):
    import os

    if silent == 0:
        print("{}".format(cmd))
    status = os.system(cmd)
    if status != 0:
        raise Exception('error when running:\n{}\n'.format(cmd))


def least_sqares(H, S, W=None):
    '''
    #This can make use multiple threads (set environment variable: OMP_NUM_THREADS)
    linear equations:  H theta = s
    W:                 weight matrix
    '''

    S.reshape(H.shape[0], 1)
    if W is None:
        #use np.dot instead since some old python versions don't have matmul
        m1 = np.linalg.inv(np.dot(H.transpose(), H))
        Z = np.dot(       np.dot(m1, H.transpose())           , S)
    else:
        #use np.dot instead since some old python versions don't have matmul
        m1 = np.linalg.inv(np.dot(np.dot(H.transpose(), W), H))
        Z = np.dot(np.dot(np.dot(m1, H.transpose()), W), S)

    return Z.reshape(Z.size)


def least_sqares_insar(H0, S0, dates2, dateZero, pairs, W0=None, mode=0):
    '''
    least squares for insar time series analysis. basic equation
    H0 dates2 = S0
    invalid values are zeros in S0.
    assuming full-rank if all pairs are valid. dateZero must be valid.
    
    input paramters
    H0:       observation matrix, 2-D numpy array
    S0:       insar values, 1-D numpy array
    dates2:   sorted dates, dateZero excluded, format: ['date1', 'date2'...]
    dateZero: reference date, format: 'date'
    pairs:    pairs, format: ['date1-date2', 'date1-date2'...]
    W0:       weighting matrix 1-D numpy array
    mode:     solution mode
              0: only do ls if all pairs are valid
              1: only do ls if all dates are valid
              2: do ls for all cases

    output parameters
    ts:       deformation values of dates2, dateZero excluded, format: 1-D numpy array with length ndate-1.
              zero values are not valid.

    different cases:
              | 1. all pairs √
    all dates |                   | 2. full rank √
              | a subset of pairs |
                                  | 3. low rank  ×

                                   | 4. full rank √
               | ref date included |
    some dates |                   | 5. low rank ×
               | 6. ref date not included ×
    '''

    #import numpy as np

    if mode not in [0, 1, 2]:
        raise Exception('unknown solution mode: {}'.format(mode))

    (npair, ndate) = H0.shape
    ndate += 1
    npairValid = np.sum(S0!=0, dtype=np.int32)

    #zero values are not valid
    ts = np.zeros(ndate-1, dtype=np.float32)

    #case 1. nothing to be done
    if npairValid == npair:
        S = S0
        H = H0
    else:
        if mode == 0:
            return ts

        #get valid dates first (dates included in valid pairs).
        #Assume zero values in ionPairs are not valid.
        dates2Valid = []
        for k in range(npair):
            if S0[k] != 0:
                for datek in pairs[k].split('-'):
                    if datek not in dates2Valid:
                        dates2Valid.append(datek)
        dates2Valid = sorted(dates2Valid)
        ndateValid = len(dates2Valid)
        if dateZero in dates2Valid:
            dateZeroIncluded = True
            dates2Valid.remove(dateZero)
        else:
            dateZeroIncluded = False

        #case: valid include all dates
        if ndateValid == ndate:
            #get valid
            H1 = H0[np.nonzero(S0!=0)]
            S1 = S0[np.nonzero(S0!=0)]
            #case 2. full rank
            #case 3. low rank
            if np.linalg.matrix_rank(H1) < ndate-1:
                return ts
            S = S1
            H = H1
        else:
            if mode <= 1:
                return ts

            if dateZeroIncluded:
                dates2ValidIndex = [dates2.index(x) for x in dates2Valid]
                H1 = H0[np.nonzero(S0!=0)]
                S1 = S0[np.nonzero(S0!=0)]
                H2 = H1[:, dates2ValidIndex]
                #case 4. full rank
                #case 5. low rank
                if np.linalg.matrix_rank(H2) < ndateValid-1:
                    return ts
                S = S1
                H = H2
            else:
                #case 6. reference date not in valid
                return ts

    #adding weight
    #https://stackoverflow.com/questions/19624997/understanding-scipys-least-square-function-with-irls
    #https://stackoverflow.com/questions/27128688/how-to-use-least-squares-with-weight-matrix-in-python
    if W0 is not None:
        if npairValid == npair:
            W = W0
        else:
            W = W0[np.nonzero(S0!=0)]
        H = H0 * W[:, None]
        S = S0 * W

    #do least-squares estimation
    #[theta, residuals, rank, singular] = np.linalg.lstsq(H, S)
    #make W full matrix if use W here (which is a slower method)
    #'using W before this' is faster
    theta = least_sqares(H, S, W=None)

    if npairValid == npair:
        ts = theta
    else:
        if ndateValid == ndate:
            ts = theta
        else:
            ts[dates2ValidIndex] = theta

    return ts


def gaussian_filtering(data, dt, sigma, trunc=True):
    '''
    data:    original time series (ndate, length, width)
    dt:      acquistion time in years, zero year can be any acquistion
    sigma:   Gaussian temporal filter length or sigma in years
    trunc:   whether truncate filter using filter length.
    '''

    ndate = data.shape[0]
    length = data.shape[1]
    width = data.shape[2]

    #dt2 in years
    dt2 = np.zeros((ndate, ndate))
    for i in range(ndate):
        for j in range(ndate):
            dt2[i, j] = dt[j] - dt[i]

    #generate gaussian filter for each date (ndate*ndate)
    gfilter = np.exp(-dt2**2/(2.0*sigma**2)) / (sigma * np.sqrt(2.0*np.pi))

    #if use a truncated filter, do the following
    if trunc == True:
        filter_length = sigma
        gfilter[np.nonzero(np.absolute(dt2)>filter_length/2.0)] = 0.0

    #scale
    norm = np.matlib.repmat(np.dot(gfilter, np.ones((ndate, 1))), 1, ndate)
    gfilter /= norm

    #do filtering
    est_filt = np.zeros((ndate, length, width), dtype=np.float32)
    flag = np.sum(data==0, axis=0)
    for i in range(length):
        if (i+1) % 50 == 0 or (i+1) == length:
            print('processing line: %6d of %6d' % (i+1, length), end='\r')
        if (i+1) == length:
            print()
        for j in range(width):
            if flag[i, j] != ndate:
                est_filt[:, i, j] = np.dot(gfilter, data[:, i, j]) 

    return est_filt


def linear_velocity(data, dt):
    '''
    data:     time series (ndate, length, width)
    dt:       acquistion time in years, zero year can be any acquistion
    '''

    ndate = data.shape[0]
    length = data.shape[1]
    width = data.shape[2]

    vel_mean = np.zeros((length, width))
    flag = np.sum(data==0, axis=0)
    for i in range(length):
        if (i+1) % 50 == 0 or (i+1) == length:
            print('processing line: %6d of %6d' % (i+1, length), end='\r')
        if (i+1) == length:
            print()
        for j in range(width):
            if flag[i, j] != ndate:
                p = np.polyfit(dt, data[:,i,j], 1)
                #p(x) = p[0] * x**1 + p[1]
                vel_mean[i, j] = p[0]

    return vel_mean


def form_temporal_matrix(dt, cycle):
    '''
    dt:      acquistion time in years, zero year can be any acquistion
    cycle:   seasonal cycle in years

    temporal behaviour: f(t) = b + kt + z1*cos[2pi(t/cycle)]
                                      + z1*sin[2pi(t/cycle)]
    '''

    ndate = len(dt)
    npar  = 4

    #dt2 in years, time of first acquistion is zero
    dt2 = np.zeros(ndate)
    for i in range(ndate):
            dt2[i] = dt[i] - dt[0]

    H = np.zeros((ndate, npar))
    for i in range(ndate):
        H[i, 0] = 1.0
        H[i, 1] = dt2[i]
        H[i, 2] = np.cos(2.0*np.pi*(dt2[i]/cycle))
        H[i, 3] = np.sin(2.0*np.pi*(dt2[i]/cycle))

    return H


def estimate_temporal_par(data, dt, sigma, cycle):
    '''
    data:    original time series (ndate, length, width)
    dt:      acquistion time in years, zero year can be any acquistion
    sigma:   Gaussian temporal filter length or sigma in years
    cycle:   seasonal cycle in years

    temporal behaviour: f(t) = b + kt + z1*cos[2pi(t/cycle)]
                                      + z1*sin[2pi(t/cycle)]
    '''

    ndate = data.shape[0]
    length = data.shape[1]
    width = data.shape[2]

    #first filter the data
    data_filt = gaussian_filtering(data, dt, sigma, trunc=False)

    #form matrix
    H = form_temporal_matrix(dt, cycle)

    #removing seasonal
    data_rm_temp = np.zeros((ndate, length, width), dtype=np.float32)
    flag = np.sum(data==0, axis=0)
    for i in range(length):
        if (i+1) % 50 == 0 or (i+1) == length:
            print('processing line: %6d of %6d' % (i+1, length), end='\r')
        if (i+1) == length:
            print()
        for j in range(width):
            if flag[i, j] != ndate:
                #[x, residuals, rank, s] = np.linalg.lstsq(H, data_filt[:, i, j])
                x = least_sqares(H, data_filt[:, i, j], W=None)
                data_rm_temp[:, i, j] = data[:, i, j] - np.dot(     H, np.array([0, 0, x[2], x[3]])     )

    return data_rm_temp


def create_imgs(dates, data, imgdir='daily_def', resize=50, ipl=5):
    '''
    dates:    data acquistion dates
    data:     original time series (ndate, length, width)
    imgdir:   image directory
    resize:   percentage of original img size
    ipl:      images per line
    '''
    ndate = data.shape[0]
    length = data.shape[1]
    width = data.shape[2]

    if not os.path.isdir(imgdir):
        os.makedirs(imgdir)

    mean = np.mean(data, axis=0, dtype=np.float64)
    flag = (np.sum(data, axis=0, dtype=np.float64) != 0)
    flagfilename = os.path.join(imgdir, 'flag')
    flag.astype(np.float32).tofile(flagfilename)

    #create images
    tmpfilename = os.path.join(imgdir, 'tmp')
    for i in range(ndate):
        datax = (data[i, :, :] - mean) * flag
        datax.astype(np.float32).tofile(tmpfilename)
        cmd = 'mdx {} -s {} -r4 -percent 100      {} -s {} -r4 -wrap 40 -addr -20 -cmap CMY -P -workdir {} > /dev/null'.format(
            flagfilename,
            width,
            tmpfilename,
            width,
            imgdir)
        runCmd(cmd, silent=1)
        cmd = 'convert {} -resize {}% {}'.format(
            os.path.join(imgdir, 'out.ppm'),
            resize,
            os.path.join(imgdir, dates[i] + '.tiff'))
        runCmd(cmd, silent=1)
        #remove temporary files associated with this image
        os.remove(tmpfilename)
        os.remove(os.path.join(imgdir, 'out.ppm'))

    #create svg file
    hdr = '''<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" 
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="20cm" height="20cm" version="1.1"
     xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
'''
    tlr = '''
</svg>'''
    svg = hdr
    #add images
    imgwidth = 1.2*(width*resize/100)/41
    imglength = 3.9*(length*resize/100)/120
    fontsize = 13.55*(width*resize/100)/41 * 0.3
    for i in range(ndate):
        #line and column indexes, indexes start from 1
        ii = int((i + 1 - 0.1) / ipl) + 1
        jj = i + 1 - (ii - 1) * ipl
        #create img
        img = '''    <image xlink:href="{}" x="{}cm" y="{}cm"/>
    <text x="{}cm" y="{}cm" style="font-family:'Times New Roman';font-weight:normal;font-style:normal;font-stretch:normal;font-variant:normal;font-size:{}px">{}</text>
'''.format(os.path.join(imgdir, dates[i] + '.tiff'),
           0+(jj-1)*imgwidth,
           0+(ii-1)*imglength,
           0+(jj-1)*imgwidth,
           0+(ii-1)*imglength-0.31,
           fontsize,
           dates[i]
        )
        svg += img
    svg += tlr
    #output image
    with open(imgdir+'.svg', 'w') as f:
        f.write(svg)


#######################################################
#these functions are for doing test
def select_pixels(length, width):
    '''
    length: file length
    width: file width
    '''

    indexes = (np.array([7222, 1962, 2683, 2821, 1759, 4156, 4800, 2463, 6084, 6579]), np.array([3263, 817, 1944, 715, 3631, 1761, 2977, 3199, 2644, 4747]))
    msk = np.zeros((length, width), dtype=np.float32)
    msk[indexes] = 1.0

    return msk


def delete_dates(data):
    dates_del = ['150506', 
                 '150903', 
                 '151021', 
                 '160430', 
                 '160921', 
                 '170326', 
                 '170513', 
                 '170712', 
                 '170724', 
                 '170805', 
                 '170910']
    dates_del = sorted(dates_del)
    indexes_del = []
    for x in dates_del:
        indexes_del.append(dates.index(x))

    #delete element
    data = np.delete(data, indexes_del, axis=0)
#######################################################


def cmdLineParse():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser( description='A program for least squares estimation of the InSAR time series')
    parser.add_argument('-plist', dest='plist', type=str, required=True,
            help = 'file containing the list of insar pairs. format: mdate-sdate. a line is considered as valid if it contains -')
    parser.add_argument('-unw', dest='unw', type=str, required=True,
            help = 'unwrapped interferogram files in rmg format with wildcard')
    parser.add_argument('-cor', dest='cor', type=str, default=None,
            help = 'coherence files in rmg format with wildcard. default: None')
    parser.add_argument('-ts_raw', dest='ts_raw', type=str, default=None,
            help = 'raw time series directly from least squares without any further processing. if provided, no file reading and leas squares will be done. default: None')
    parser.add_argument('-nlks', dest='nlks', type=float, default=None,
            help='effective number of looks, must be provided when -cor is provided')
    parser.add_argument('-std', dest='std', type=str, default=None,
            help = 'standard deviation files in single band float format with wildcard, higher priority over coherence. default: None.')
    parser.add_argument('-tspan', dest='tspan', action='store_true', default=False,
            help='consider pair time span in the weight of least squares')
    parser.add_argument('-output', dest='output', type=str, required=True,
            help = 'output mean velocity file')
    parser.add_argument('-wvl', dest='wvl', type=float, required=True,
            help = 'radar wave length in m. the output velocity is in mm/year.')
    parser.add_argument('-sigma', dest='sigma', type=float, default=0.8,
            help = 'Gaussian temporal filter sigma (length) in years. default: 0.8')
    parser.add_argument('-vel_source', dest='vel_source', type=int, default=0,
            help = 'mean velocity estimated using 0: raw time series (default). 1: temporally filtered time series.')
    parser.add_argument('-bbox', nargs=4, dest='bbox', type=int, default=None,
            help = 'reference region, format: fl ll fc lc. default: 1 length 1 width (whole image).')
    parser.add_argument('-do_not_rm_const', dest='do_not_rm_const', action='store_true', default=False,
            help='do not remove a constant from the interferogram')
    parser.add_argument('-seasonal', dest='seasonal', type=int, default=0,
            help = 'whether remove seasonal signal. 0: no (default). 1: yes.')
    parser.add_argument('-mode', dest='mode', type=int, default=1,
            help = 'least squares solution mode. 0: only do ls if all pairs are valid. 1: only do ls if all dates are valid (default). 2: do ls for all cases. invalid values are zeros in the interferograms.')
    parser.add_argument('-errors', dest='errors', type=str, nargs='+', default=None,
            help = 'error sources in radians to be removed. this is a list of directories, with each directory containing a kind of error. Each directory contains error files of all dates that can be sorted chronologically by python sort function')
    parser.add_argument('-error_coeffs', dest='error_coeffs', type=float, nargs='+', default=None,
            help = 'these are coefficients that should be added to the errors')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()


####################################################################################
    print('\nSTEP 1. prepare for processing')
####################################################################################
    #get dates and pairs
    dates = [] #all dates including the first one.
    pairs = [] #all pairs
    with open(inps.plist) as f:
        lines = f.readlines()
    for line_i in lines:
        if '-' in line_i:
            pairs.append(line_i.strip().strip('\n'))
        else:
            continue
        mdate = line_i.split('-')[0].strip().strip('\n')
        sdate = line_i.split('-')[1].strip().strip('\n')
        if mdate not in dates:
            dates.append(mdate)
        if sdate not in dates:
            dates.append(sdate)
    dates = sorted(dates)
    pairs = sorted(pairs)
    ndate = len(dates)
    npair = len(pairs)

    #convert dates to python datetime format
    dates2 = []
    for i in range(ndate):
        dates2.append(datetime.datetime(2000+int(dates[i][0:2]), int(dates[i][2:4]), int(dates[i][4:6]), 00, 00))
    #acquistion time in years, first acquisiton is zero
    dt = np.zeros(ndate)
    for i in range(ndate):
        dt[i] = (dates2[i] - dates2[0]).total_seconds() / (365.0 * 24 * 60 * 60)

    dt_pairs = []
    for i in range(npair):
        mdate, sdate = pairs[i].split('-')
        mdate0 = datetime.datetime(2000+int(mdate[0:2]), int(mdate[2:4]), int(mdate[4:6]), 00, 00)
        sdate0 = datetime.datetime(2000+int(sdate[0:2]), int(sdate[2:4]), int(sdate[4:6]), 00, 00)
        dt_pairs.append(abs((sdate0 - mdate0).total_seconds() / (365.0 * 24 * 60 * 60)))

    #get unwrapped interferogram and coherence files
    #make sure this is in the same order as pairs
    unwf = sorted(glob.glob(inps.unw))
    print(len(unwf),npair)

    if not (len(unwf) == npair):
        raise Exception('not same number of files\n')
    if inps.cor is not None:
        if inps.nlks is None:
            raise Exception('effective number of looks -nlks must be provided when -cor is provided\n')
        corf = sorted(glob.glob(inps.cor))
        if not (len(corf) == npair):
            raise Exception('not same number of files\n')

        xmlx = ElementTree(file=corf[0] + '.xml').getroot()
        tmp = xmlx.find("property[@name='number_bands']/value")
        nbands_cor = int(tmp.text)

    if inps.std is not None:
        stdf = sorted(glob.glob(inps.std))
        if not (len(stdf) == npair):
            raise Exception('not same number of files\n')

    if inps.errors is not None:
        if inps.error_coeffs is None:
            raise Exception('errors are provided, but error_coeffs are not')
        else:
            if len(inps.error_coeffs) != len(inps.errors):
                raise Exception('number of errors are not equal to number of error_coeffs')
        for x in inps.errors:
            xf = sorted(glob.glob(x))
            if not (len(xf) == ndate):
                raise Exception('not same number of error files\n')

    #get width and length from header file of first interferogram
    xmlx = ElementTree(file=unwf[0] + '.xml').getroot()
    tmp = xmlx.find("component[@name='coordinate1']/property[@name='size']/value")
    width = int(tmp.text)
    tmp = xmlx.find("component[@name='coordinate2']/property[@name='size']/value")
    length = int(tmp.text)
    tmp = xmlx.find("property[@name='number_bands']/value")
    nbands_unw = int(tmp.text)

    #get reference region
    if inps.bbox == None:
        bbox = [1, length, 1, width]
    else:
        bbox = inps.bbox

    if not (1<=bbox[0]<=length and 1<=bbox[1]<=length and 1<=bbox[2]<=width and 1<=bbox[3]<=width):
        raise Exception("wrong reference region!")
    else:
        print('The reference region you specified is:')
        print('first line:   {}'.format(bbox[0]))
        print('last line:    {}'.format(bbox[1]))
        print('first column: {}'.format(bbox[2]))
        print('last column:  {}'.format(bbox[3]))
    bbox = [x-1 for x in bbox]

    #convert rad to mm
    wvl = inps.wvl * 1000.0

    meta_dict = {'width': width,
                 'length': length,
                 'dates': dates2,
                 'dt':dt }

    #dump meta data
    file_name = 'meta.pck'
    with open(file_name, 'wb') as f:
        pickle.dump(meta_dict, f)


####################################################################################
    print('\nSTEP 2. read files')
####################################################################################
    if inps.ts_raw is not None:
        print('raw time series already provided. skip reading files')
    else:
        unw = np.zeros((npair, length, width), dtype=np.float32)
        if (inps.cor is not None) or (inps.std is not None):
            wgt = np.zeros((npair, length, width), dtype=np.float32)
        for i in range(npair):
            print('reading file: %3d of %3d' % (i+1, npair), end='\r')
            if i+1 == npair:
                print()

            if nbands_unw == 1:
                #for RMG format
                unw[i, :, :] = np.fromfile(unwf[i], dtype=np.float32).reshape(length, width)
            elif nbands_unw == 2:
                #single band float format
                unw[i, :, :] = (np.fromfile(unwf[i], dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
            else:
                raise Exception("unw format not supported")
            #remove constant phase
            flag = (unw[i, :, :]!=0)
            ref_region = unw[i, bbox[0]:bbox[1], bbox[2]:bbox[3]]
            if not inps.do_not_rm_const:
                #only remove if there are valid values in ref region
                if len(np.nonzero(ref_region!=0)[0]) != 0:
                    unw[i, :, :] -= flag * np.mean(ref_region[np.nonzero(ref_region!=0)], dtype=np.float64)
                else:
                    print('WARNING: there are no valid pixels in reference region for interferogram {}'.format(i+1))

            if (inps.cor is not None) and (inps.std is None):
                if nbands_cor == 1:
                    #for single band float format
                    cor = np.fromfile(corf[i], dtype=np.float32).reshape(length, width)
                elif nbands_cor == 2:
                    #for RMG format
                    cor = (np.fromfile(corf[i], dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
                else:
                    raise Exception("cor format not supported")
                #remove odd values
                #cor[np.nonzero(cor<0)] = 0.0
                #cor[np.nonzero(cor>=0.99)] = 0.0
                cor[np.nonzero(cor<0)] = 0.01
                cor[np.nonzero(cor>=0.99)] = 0.99
                #var[i, :, :] = (cor!=0) * (1.0 - cor**2) / (cor + (cor==0))**2 / (2.0 * inps.nlks)
                wgt[i, :, :] = 2.0 * inps.nlks * (cor**2) / (1.0 - cor**2)
                if inps.tspan:
                    wgt[i, :, :] *= (dt_pairs[i]**2)

            if inps.std is not None:
                std = np.fromfile(stdf[i], dtype=np.float32).reshape(length, width)
                wgt[i, :, :] =  (1.0 / (std +(std==0)))**2
                if inps.tspan:
                    wgt[i, :, :] *= (dt_pairs[i]**2)

        unw *= (wvl / (4.0 * np.pi))
        unw_valid = np.count_nonzero(unw, axis=0)

        if (inps.cor is not None) or (inps.std is not None):
            #use coherece as weight directly
            #wgt = cor
            wgt_valid = np.count_nonzero(wgt, axis=0)


####################################################################################
    print('\nSTEP 3. do least squares')
####################################################################################
    if inps.ts_raw is not None:
        print('raw time series already provided. skip least squares')
        ts = np.fromfile(inps.ts_raw, dtype=np.float32).reshape(ndate, length, width)
    else:
        ###################################################
        #FOR DOING TEST: select pixels to process
        if 0:
            unw[0, :, :] *= select_pixels(length, width)
        ###################################################

        #observation matrix
        H0 = np.zeros((npair, ndate-1))
        for k in range(npair):
            mdate = pairs[k].split('-')[0]
            sdate = pairs[k].split('-')[1]
            mdate_i = dates.index(mdate)
            sdate_i = dates.index(sdate)
            if mdate_i != 0:
                H0[k, mdate_i-1] = 1
            if sdate_i != 0:
                H0[k, sdate_i-1] = -1

        ts = np.zeros((ndate, length, width), dtype=np.float32)
        for i in range(length):
            if (i+1) % 50 == 0 or (i+1) == length:
                print('processing line: %6d of %6d' % (i+1, length), end='\r')
            if (i+1) == length:
                print()
            for j in range(width):
                if (inps.cor is None) and (inps.std is None):
                    W = None
                else:
                    W = np.sqrt(wgt[:, i, j])

                #whether doing least squares are determined by the zeros (invalid values) in unw.
                ts[1:, i, j] = least_sqares_insar(H0, unw[:, i, j], dates[1:], dates[0], pairs, W0=W, mode=inps.mode)

    #create quick-look image for each date
    if 0:
        print('create quick-look images')
        create_imgs(dates, ts)

    ###################################################
    #FOR DOING TEST: remove some dates
    if 0:
        delete_dates(dt)
        delete_dates(ts)
    ###################################################

    #remove error sources
    if inps.errors is not None:
        print('remove errors')
        nerror = len(inps.errors)
        for i in range(nerror):
            print('remove error {} of {}'.format(i+1, nerror))
            error_files = sorted(glob.glob(inps.errors[i]))
            error_ref = (wvl / (4.0 * np.pi)) * np.fromfile(error_files[0], dtype=np.float32).reshape(length, width)
            for j in range(ndate):
                #in ts
                #the first date (date_0) is reference. deformation of all date_j are date_j - date_0.
                #so deformation of each date is also a pair: date_j-date_0
                #when the phase is converted to deformation, sign is not changed.
                #so the sign of the error (inps.error_coeffs[i]) is the same as that of a regular InSAR pair
                #(date_j-date_0) + (sign) * (date_error_j-date_error_0)
                error_sec = (wvl / (4.0 * np.pi)) * np.fromfile(error_files[j], dtype=np.float32).reshape(length, width)
                #error_pair = (error_sec != 0) * (error_ref != 0) * (error_sec - error_ref)
                error_pair = error_sec - error_ref
                #remove reference region values
                if not inps.do_not_rm_const:
                    ref_region = error_pair[bbox[0]:bbox[1], bbox[2]:bbox[3]]
                    #only remove if there are valid values in ref region
                    if len(np.nonzero(ref_region!=0)[0]) != 0:
                        error_pair -= (error_pair!=0) * np.mean(ref_region[np.nonzero(ref_region!=0)], dtype=np.float64)
                    else:
                        if j != 0:
                            print('WARNING: there are no valid pixels in reference region for date {} of error source {}'.format(dates[j], i+1))
                #assuming error_pair cover all valid ts[j, :, :] pixels
                ts[j, :, :] += (ts[j, :, :] != 0) * error_pair * inps.error_coeffs[i]

    #dump raw estimate
    file_name = 'ts_raw.dat'
    ts.astype(np.float32).tofile(file_name)


####################################################################################
    print('\nSTEP 4. remove seasonal deformation')
####################################################################################
    if inps.seasonal == 1:
        ts_rm_temp = estimate_temporal_par(ts, dt, 0.2, 1.0)
    else:
        print('seasonal deformation not removed')
        ts_rm_temp = ts


####################################################################################
    print('\nSTEP 5. do Gaussian filtering')
####################################################################################
    ts_filt = gaussian_filtering(ts_rm_temp, dt, inps.sigma, False)

    #dump filt estimate
    file_name = 'ts_filt.dat'
    ts_filt.astype(np.float32).tofile(file_name)


####################################################################################
    print('\nSTEP 6. fit a 1st order polynomial to get mean velocity')
####################################################################################
    if inps.vel_source == 0:
        print('estimating mean velocity using raw time series')
        vel_mean = linear_velocity(ts_rm_temp, dt)
    else:
        print('estimating mean velocity using temporally filtered time series')
        vel_mean = linear_velocity(ts_filt, dt)


####################################################################################
    print('\nSTEP 7. write result')
####################################################################################
    #write result in rmg format
    if nbands_unw == 2:
        amp = (np.fromfile(unwf[0], dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]
    else:
        amp = (vel_mean!=0)+0
    flag = (amp!=0)*(vel_mean!=0)
    img = np.zeros((length*2,width), dtype=np.float32)
    img[0:length*2:2, :] = amp*flag
    img[1:length*2:2, :] = vel_mean*flag
    img.astype(np.float32).tofile(inps.output)

    #create_xml(inps.output, width, length, 'rmg'):
    try:
        ##create xml, need isce
        #import isce, isceobj
        #img = isceobj.createImage()
        #img.load(unwf[0]+'.xml')
        #img.setFilename(inps.output)
        #img.extraFilename = inps.output+'.vrt'
        #img.renderHdr()

        import isce, isceobj
        from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
        create_xml(inps.output, width, length, 'rmg')
    except ImportError:
        print('ISCE not available')
        shutil.copy(unwf[0]+'.xml', '{}.xml'.format(inps.output))
