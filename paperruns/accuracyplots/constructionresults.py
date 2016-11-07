
# November 2016: Results on construction test set

# dict elements are:
# tot: Total accuracy (percent)
# pos: True positive rate (percent)
# neg: True negative rate (percent)
# tpprop: True positives divided by total detections
# fd: Number of false detections 
# tp: Number of true positives
# (total detections = true positives + false detections)

import numpy as np

# KAZE-kNN5-nbhd30-thr1e-XX-ctfrac0.01ctnbhd120
thresholds = [1e-02,3e-03,1e-03,3e-04,1e-04,3e-05,1e-05,3e-06,1e-06,3e-07,
              1e-07,3e-08,1e-08,3e-09,1e-09,3e-10,1e-10]
logthresh = [np.log10(x) for x in thresholds]
KAZEkNN5 = dict(
    tot = np.array(zip(logthresh,100*np.array(
     [.490,.520,.640,.670,.680,.640,.670,.640,.600,.600,
      .570,.560,.560,.550,.530,.520,.510]))),
    pos = np.array(zip(logthresh,100*np.array(
     [.94,.92,.9,.82,.74,.6,.56,.44,.28,.24,.18,.14,.12,.1,.06,.04,.02]))),
    neg = np.array(zip(logthresh,100*np.array(
     [.04,.12,.38,.52,.62,.68,.78,.84,.92,.96,.96,.98,1.0,1.0,1.0,
      1.0,1.]))),
    tpprop = np.array(zip(logthresh,np.array(
     [47/96.,46/92.,45/79.,41/68.,37/61.,30/53.,28/45.,22/35.,14/23.,12/16.,
      9/13.,7/8.,6/6.,5/6.,3/4.,2/3.,1/2.]))),
    fd = np.array(zip(logthresh,
     np.array([49.,46.,34.,27.,24.,23.,17.,13.,9.,4.,4.,1.,0.,1.,1.,1.,1.])))
    )
KAZEkNN5['tp'] = np.array(zip(logthresh,KAZEkNN5['pos'][:,1]*.50))
KAZEkNN5['thresh'] = logthresh

# KAZE-kNN1-nbhd30-thr1e-XX-ctfrac0.01ctnbhd120
thresholds = [1e-02,3e-03,1e-03,3e-04,1e-04,3e-05,1e-05,3e-06,1e-06,
              3e-07,1e-07,3e-08,1e-08,3e-09,1e-09,3e-10,1e-10]
logthresh = [np.log10(x) for x in thresholds]
KAZEkNN1 = dict(
    tot = np.array(zip(logthresh,100*np.array(
     [.480,.530,.600,.630,.620,.630,.560,.570,.570,.580,.580,.540,
      .540,.520,.520,.520,.520]))),
    pos = np.array(zip(logthresh,100*np.array(
     [.96,.9,.88,.82,.72,.66,.46,.34,.24,.22,.18,.1,.08,.04,.04,.04,.04]))),
    neg = np.array(zip(logthresh,100*np.array(
     [0.0,.16,.32,.44,.52,.6,.66,.8,.9,.94,.98,.98,1.0,1.0,1.0,1.0,1.0]))),
    tpprop = np.array(zip(logthresh,np.array(
     [48/98.,45/89.,44/81.,41/71.,36/65.,33/59.,23/48.,17/31.,12/20.,11/16.,
      9/11.,5/8.,4/6.,2/3.,2/3.,2/2.,2/2.]))),
    fd = np.array(zip(logthresh,
            np.array([50.,44.,37.,30.,29.,26.,15.,14.,8.,5.,2.,3.,
                      2.,1.,1.,0.,0.])))
    #tpfp = np.array(zip(logthresh,1-1/np.array(
    #[.959,1.0,1.324,1.519,1.542,1.304,1.647,1.692,1.556,2.25,INF,3.0,1.0])))
    )
KAZEkNN1['tp'] = np.array(zip(logthresh,KAZEkNN1['pos'][:,1]*.50))
KAZEkNN1['thresh'] = logthresh

# SIFT-kNN5-nbhd30-thr1e-XX-ctfrac0.01ctnbhd120
thresholds = [1e-01,1e-02,1e-03,1e-04,1e-05,1e-06,1e-07,1e-08]
logthresh = [np.log10(x) for x in thresholds]
SIFTkNN5 = dict(
    tot = np.array(zip(logthresh,100*np.array(
     [.480,.550,.630,.550,.500,.500]))),
    pos = np.array(zip(logthresh,100*np.array(
     [.96,.74,.42,.12,.02,0.0,]))),
    neg = np.array(zip(logthresh,100*np.array(
     [0.0,.36,.84,.98,.98,1.0,]))),
    tpprop = np.array(zip(logthresh,np.array(
     [48/99.,37/74.,21/33.,6/9.,1/2.,None]))),
    fd = np.array(zip(logthresh,
            np.array([51.,37.,12.,3.,1.,0.])))
    )
SIFTkNN5['tp'] = np.array(zip(logthresh,SIFTkNN5['pos'][:,1]*.50))
SIFTkNN5['thresh'] = logthresh

# SIFT-kNN1-nbhd30-thr1e-XX-ctfrac0.01ctnbhd120
thresholds = [3e-01,1e-01,3e-02,1e-02,3e-03,1e-03,3e-04,1e-04,1e-05]
logthresh = [np.log10(x) for x in thresholds]
SIFTkNN1 = dict(
    tot = np.array(zip(logthresh,100*np.array(
     [.490,.480,.490,.510,.560,.520,.490,.500,.500]))),
    pos = np.array(zip(logthresh,100*np.array(
     [.98,.94,.86,.66,.4,.18,.08,.02,0.0]))),
    neg = np.array(zip(logthresh,100*np.array(
     [0.0,.02,.12,.36,.72,.86,.9,.98,1.0]))),
    tpprop = np.array(zip(logthresh,np.array(
     [49/99.,47/96.,43/88.,33/69.,20/41.,9/20.,4/13.,1/2.,0.]))),
    fd = np.array(zip(logthresh,
            np.array([50.,49.,45.,36.,21.,11.,9.,1.,1.])))
    #tpfp = np.array(zip(logthresh,1-1/np.array(
    # [.98,.959,0,.917,.952,.818,.444,.5,0.0])))
    )
SIFTkNN1['tp'] = np.array(zip(logthresh,SIFTkNN1['pos'][:,1]*.50))
SIFTkNN1['thresh'] = logthresh

