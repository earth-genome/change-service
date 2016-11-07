
# 11/16 - plotting routines for results on construction test set

# output data is recorded in dicts in constructionresults.py

# 11/4/16: fd, tp, and tpfdplot(), tpplot(), came from aborted attempts to
# plot the difference (true positives - false detections) or number of
# true positives along with the proportion of true positives (tpprop)
# out of total detections.


import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb

import constructionresults as cr

dataruns = [cr.KAZEkNN5,cr.KAZEkNN1,cr.SIFTkNN5,cr.SIFTkNN1]
totalaccuracy = [[d['tot'] for d in dataruns]]
posratenegrate = [[d['pos'] for d in dataruns],[d['neg'] for d in dataruns]]
tpprop = [[d['tpprop'] for d in dataruns]]
#tpfd = zip([d['tp'] for d in dataruns],[d['fd'] for d in dataruns])
minthresh = min([t for d in dataruns for t in d['thresh']])
maxthresh = max([t for d in dataruns for t in d['thresh']])


def commonplot(ax,data,savename,title,xlabel,ylabel,xlim=None,ylim=None):
    """Create an ad-hoc 2D plot of assembled data."""
    colors =  ['b','g','r','k']
    markers = ['^','s','o','*']
    #markers = [None for _ in dataruns]
    #pdb.set_trace()
    for trio in data:
        for d,(c,m) in zip(trio,zip(colors,markers)):
            ax.plot(d[:,0],d[:,1],color=c,marker=m,fillstyle='none')
    #ax.set_xscale('log')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    if xlabel is not None:
        ax.set_xlabel(r"Probability threshold ($\log\,\tilde\varepsilon$)",
                  fontsize=20)
    ax.set_title(title,fontsize=20)
    plt.savefig(savename,bbox_inches='tight')

def tpfdplot(ax,data,tpfd,savename,title,xlabel,ylabel,xlim=None,ylim=None):
    """Create an ad-hoc 2D plot of assembled data."""
    colors =  ['b','g','r','k']
    markers = ['^','s','o','*']
    #markers = [None for _ in dataruns]
    for trio in data:
        for d,(c,m) in zip(trio,zip(colors,markers)):
            ax.plot(d[:,0],d[:,1],color=c,marker=m,fillstyle='none')
    #ax.set_xscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    if xlabel is not None:
        ax.set_xlabel(r"Probability threshold ($\log\,\tilde\varepsilon$)",
                  fontsize=20)
    ax.set_title(title,fontsize=20)

    ax2 = ax.twinx()
    for (tp,fd),(c,m) in zip(tpfd,zip(colors,markers)):
        x = tp[:,0]
        y1 = tp[:,1]
        y2 = fd[:,1]
        ax2.plot(x,y1,color=c,marker=m,fillstyle='none')
        ax2.plot(x,y2,color=c,marker=m,fillstyle='none')
        ax2.fill_between(x, y1, y2, where=y2 >= y1,
                             facecolor=c, hatch='||')
        ax2.fill_between(x, y1, y2, where=y2 <= y1,
                             facecolor=c, hatch=None)
    plt.savefig(savename,bbox_inches='tight')

def tpplot(ax,data,tpfd,savename,title,xlabel,ylabel,xlim=None,ylim=None):
    """Create an ad-hoc 2D plot of assembled data."""
    colors =  ['b','g','r','k']
    markers = ['^','s','o','*']
    #markers = [None for _ in dataruns]
    for trio in data:
        for d,(c,m) in zip(trio,zip(colors,markers)):
            ax.plot(d[:,0],d[:,1],color=c,marker=m,fillstyle='none')
    #ax.set_xscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    if xlabel is not None:
        ax.set_xlabel(r"Probability threshold ($\log\,\tilde\varepsilon$)",
                  fontsize=20)
    ax.set_title(title,fontsize=20)

    ax2 = ax.twinx()
    bw = .2
    #bw = .35
    offsets = [-1.5*bw,-.5*bw,.5*bw,1.5*bw]
    #offsets = [0 for _ in range(len(colors))]
    for (tp,fd),(c,o) in zip(tpfd,zip(colors,offsets)):
        ax2.bar(tp[:,0]+o,tp[:,1],width=bw,color=c)
    ax2.set_ylim([0,150])
    ax2.set_yticks([0,10,20,30])
    ax2.set_ylabel('True positives',fontsize=20)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    #if xlim is not None:
    #    ax2.set_xlim(xlim)
    plt.savefig(savename,bbox_inches='tight')

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    commonplot(ax,totalaccuracy,savename='TA.png',
               title='Total accuracy',
               xlabel=None,ylabel='%',xlim=[maxthresh,minthresh],
               ylim=[40,80])
    ax.legend(['KAZE-kNN5','KAZE-kNN1','SIFT-kNN5','SIFT-kNN1'],loc=1)
    plt.savefig('TA.png',bbox_inches='tight')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    commonplot(ax,posratenegrate,savename='posneg.png',
               title='True positive and true negative rates',
               xlabel=True,ylabel='%',xlim=[maxthresh,minthresh],
               ylim=[0,100])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #tpfdplot(ax,tpprop,tpfd,savename='tpprop.png',
    #           title='True positives vs. total detections',
    #           xlabel=None,ylabel=None,xlim=[maxthresh,minthresh],
    #           ylim=[0,1])
    commonplot(ax,tpprop,savename='tpprop.png',
               title='True positive proportion of total detections',
               xlabel=None,ylabel=None,xlim=[maxthresh,minthresh],
               ylim=[0,1])
    #tpplot(ax,tpprop,tpfd,savename='tpprop.png',
    #           title='True positive proportion of total detections',
    #           xlabel=None,ylabel=None,xlim=[maxthresh,minthresh],
    #           ylim=[0,1])
    TApng = plt.imread('TA.png')
    posnegpng = plt.imread('posneg.png')
    tpproppng = plt.imread('tpprop.png')
    print TApng.shape, posnegpng.shape, tpproppng.shape
    labelheight = posnegpng.shape[0] - TApng.shape[0]
    print np.max(TApng)
    TApng = np.vstack([TApng,np.ones((labelheight,TApng.shape[1],4))])
    tpproppng = np.vstack([tpproppng,
                         np.ones((labelheight,tpproppng.shape[1],4))])
    trip = np.hstack([TApng,posnegpng,tpproppng])
    plt.imsave('triptych-contstruction.png',trip)

    

    
