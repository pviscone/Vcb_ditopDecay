# Reconstruct the $t\bar{t}$ kinematics/powheg-hvq dataset

> 1. - [x] Understand the cuts of montecarlo:
>      
>      https://cms-pdmv.cern.ch/mcm/requests?dataset_name=TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8&page=0&shown=127
>      
>      In the option fragment there is a script in wich there is a path: 
>      
>      /cvmfs/cms.cern.ch/phys_generator/gridpacks/2017/13TeV/powheg/V2/TT_hvq/TT_hdamp_NNPDF31_NNLO_ljets.tgz
>      
>      Unpack this file on the cernbox (*huge file*) and copy only the useful info on github.
>      
>      (Go to the TWiki and try to understand how to understand how a montecarlo is generated)

---

# Info

- file: [A761E638-9C89-644F-8C33-801D58DEB328.root](https://cmsweb.cern.ch/das/request?input=file%3D%2Fstore%2Fmc%2FRunIISummer20UL17NanoAODv2%2FTTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8%2FNANOAODSIM%2F106X_mc2017_realistic_v8-v1%2F120000%2FA761E638-9C89-644F-8C33-801D58DEB328.root&instance=prod/global) (1.6GB) called ttbar.root on the local repo

---

---

# LHEPart_pdgId structure

There are 9 particles (the 6th(5) and the 7th(6) are produced by the same W. The same stans for the 8th(7) and 9th(8) ):

* The first 2 (0,1)particles of each events are the incoming particles (gluons)

* The 3rd (2) particle is the additional parton (NLO montecarlo)

* The 4th and the 5th (3,4)are the $q_-$ and $\bar{q}^{'}_+$ produced by  $t\bar{t}$

* Be aware!! The 6th (5) particle could be produced both from the $t$ or the $\bar{t}$. Look at the electric charges
  
  - IF the 6th particle has charge < 0, it comes from a $W^-$
  
  - IF the 6th particle has charge  >0, it comes from a $W^+$

The quark in $t \to q W^\pm$ has the opposite charge of the $W^\pm$ and $t$ has the same charge of the $W$ ( so $t \to q_{-}W^+$  or $\bar{t} \to \bar{q}_{+}W^-$ )

# Observation and doubts

### The Ws decay only in ud,us,cd,cs pais. The motecarlo was generated with some strange cuts???:

Probably only the Cabibbo mixing is enabled.

In [TT_hdamp_NNPDF31_NNLO_ljets/poweg.input](TT_hdamp_NNPDF31_NNLO_ljets/poweg.input) there is:

```fortran
topdecaymode 11111   ! an integer of 5 digits that are either 0, or 2, representing in 
                     ! the order the maximum number of the following particles(antiparticles)
                     ! in the final state: e  mu tau up charm
                     ! For example
                     ! 22222    All decays (up to 2 units of everything)
                     ! 20000    both top go into b l nu (with the appropriate signs)
                     ! 10011    one top goes into electron (or positron), the other into (any) hadrons,
                     !          or one top goes into charm, the other into up
                     ! 00022    Fully hadronic
                     ! 00002    Fully hadronic with two charms
                     ! 00011    Fully hadronic with a single charm
                     ! 00012    Fully hadronic with at least one charm

semileptonic 1      ! uncomment if you want to filter out only semileptonic events. For example,
                     ! with topdecaymode 10011 and semileptonic 1 you get only events with one top going
                     ! to an electron or positron, and the other into any hadron.

! Parameters for the generation of spin correlations in t tbar decays
tdec/wmass 80.4  ! W mass for top decay
tdec/wwidth 2.141
tdec/bmass 4.8
tdec/twidth  1.31 ! 1.33 using PDG LO formula
tdec/elbranching 0.108
tdec/emass 0.00051
tdec/mumass 0.1057
tdec/taumass 1.777
tdec/dmass   0.100
tdec/umass   0.100
tdec/smass   0.200
tdec/cmass   1.5
tdec/sin2cabibbo 0.051
```

```bash
runcmsgrid.sh:process="hvq" #heavy quark pair production
runcmsgrid_par.sh:process="hvq"
```

Manual powheg-hvq (see page 6 and 7):

https://mobydick.mib.infn.it/~nason/POWHEG/HeavyQuarks/Powheg-hvq-manual-1.01.pdf

In the code of the generator there is ( [POWHEG/ttdec.f at 47e17e62203a1313961e7a54d45738470344875b · alisw/POWHEG · GitHub](https://github.com/alisw/POWHEG/blob/47e17e62203a1313961e7a54d45738470344875b/hvq/ttdec.f) ) (845)

```fortran
c pdg id's of 1st and 2nd W+ decay products for e,mu,tau,up and charm decays (ignoring CKM)
      data ((iwp(j,k),k=1,2),j=1,5)/-11,12, -13,14, -15,16, -1,2, -3,4/
      save ini,probs,iwp,mass,sin2cabibbo,semileptonic
```

---

POWHEG-BOX has a lot of generators ( [Homepage of the POWHEG BOX](https://powhegbox.mib.infn.it/#NLOps) ) (the source code can be downloaded with the following command):

```bash
svn checkout --username anonymous --password anonymous svn://powhegbox.mib.infn.it/trunk/POWHEG-BOX
```

In some of these generators, e.g. powheg-st (single top), the CKM mixing is enabled ( https://virgilio.mib.infn.it/~re/POWHEG/st/POWHEG-st/st_manual_v1.0.ps.gz ) 

---

In the file [powhegCKM.txt](powhegCKM.txt) I have exported the output of 

```bash
grep -Ri "CKM" . #(folder repo of powheg-box)
```

Looking at the output you can easily understand which generator exploit the ckm mixing.

## Conclusion

The dataset is not suitable for the analisys, the CKM mixing is disabled in the montecarlo generator
