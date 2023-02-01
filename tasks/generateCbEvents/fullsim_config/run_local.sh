#! /bin/bash


here=`pwd`
echo $here
cd $CMSSW_BASE/..
cd CMSSW_10_6_30_patch1/src
cmsenv
cd $here

echo $CMSSW_BASE


cmsRun step1.py

here=`pwd`
echo $here
cd $CMSSW_BASE/..
cd CMSSW_10_6_17_patch1/src
cmsenv
cd $here

cmsRun step2.py
#rm step1.root
cmsRun step3.py
#rm step2.root

here=`pwd`
cd $CMSSW_BASE/..
cd CMSSW_10_2_16_UL/src
cmsenv
cd $here

cmsRun step4.py
#rm step3.root

here=`pwd`
cd $CMSSW_BASE/..
cd CMSSW_10_6_17_patch1/src
cmsenv
cd $here

cmsRun step5.py
#rm step4.root


here=`pwd`
cd $CMSSW_BASE/..
cd CMSSW_10_6_20/src
cmsenv
cd $here


cmsRun step6.py
#rm step5.root


here=`pwd`
cd $CMSSW_BASE/..
cd CMSSW_10_6_26/src
cmsenv
cd $here


cmsRun step7.py
#rm step6.root
