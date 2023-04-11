#! /bin/bash

mkdir custom_lib
mv libnsl.so.2 ./custom_lib/
mv libtirpc.so.3 ./custom_lib
export LD_LIBRARY_PATH=$HOME/lib:$HOME/custom_lib/:$LD_LIBRARY_PATH:/usr/lib:/usr/lib64
export PATH=$HOME/lib/:$HOME/custom_lib:$PATH

echo "checking for libnsl in /usr/lib(64)"
ls /usr/lib | grep libnsl
ls /usr/lib64 | grep libnsl

#You should start and launch this from 10_6_30_patch1

#rm -rf $CMSSW_BASE/lib/
#rm -rf $CMSSW_BASE/src/
#rm -rf $CMSSW_BASE/python/
#mv lib $CMSSW_BASE/lib
#mv src $CMSSW_BASE/src
#mv python $CMSSW_BASE/python

here=`pwd`

echo "Starting from $here"
echo `ls`

cd $HOME
echo "Now I am in the HOME: $HOME"
echo `ls`
scramv1 project CMSSW_10_6_17_patch1
scramv1 project CMSSW_10_6_20
scramv1 project CMSSW_10_2_16_UL
scramv1 project CMSSW_10_6_26

echo "Home after cmsrel"
echo `ls`

cd $CMSSW_BASE
echo "This is CMSSW_BASE: $CMSSW_BASE"
echo `ls`

echo "Going back to home"
cd $HOME

cd $HOME
cd CMSSW_10_6_30_patch1/src
eval `scramv1 runtime -sh`
scram build
cd $here



echo "Running step1"
cmsRun -j FrameworkJobReport.xml -p step1.py
echo "Step 1 finished"


cd $HOME
cd CMSSW_10_6_17_patch1/src
eval `scramv1 runtime -sh`
cd $here

cmsRun -j FrameworkJobReport.xml -p step2.py
rm step1.root
cmsRun -j FrameworkJobReport.xml -p step3.py
rm step2.root


cd $HOME
cd CMSSW_10_2_16_UL/src
eval `scramv1 runtime -sh`
cd $here

cmsRun -j FrameworkJobReport.xml -p step4.py
rm step3.root


cd $HOME
cd CMSSW_10_6_17_patch1/src
eval `scramv1 runtime -sh`
cd $here

cmsRun -j FrameworkJobReport.xml -p step5.py
rm step4.root


cd $HOME
cd CMSSW_10_6_20/src
eval `scramv1 runtime -sh`
cd $here


cmsRun -j FrameworkJobReport.xml -p step6.py
rm step5.root



cd $HOME
cd CMSSW_10_6_26/src
eval `scramv1 runtime -sh`
cd $here


cmsRun -j FrameworkJobReport.xml -p step7.py
#rm step6.root
