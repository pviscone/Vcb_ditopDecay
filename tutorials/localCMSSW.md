To install all the utilities in your personal computer:

## Configure CMSSW

1. Install cvmfs
   
   ```bash
   paru -S cvmfs
   ```

  2. Look at the final output, there are useful information to configure the fs and the auto mounting

- ```bash
  sudo mkdir -p /cvmfs/cms.cern.ch
  sudo mount -t cvmfs cms.cern.ch /cvmfs/cms.cern.ch
  
  
  echo 'CVMFS_REPOSITORIES=cms.cern.ch,grid.cern.ch,sft.cern.ch\nCVMFS_HTTP_PROXY=DIRECT' | sudo tee /etc/cvmfs/default.local
  echo 'cms.cern.ch /cvmfs/cms.cern.ch cvmfs noauto,x-systemd.automount,x-systemd.requires=network-online.target,x-systemd.idle-timeout=5min,x-systemd.device-timeout=10,_netdev 0 0' | sudo tee -a /etc/fstab
  echo 'user_allow_other' | sudo tee -a /etc/fuse.conf
  
  
  
  ```



3. Add to .zshrc:
   
   ```bash
   source /cvmfs/cms.cern.ch/cmsset_default.sh
   source /cvmfs/cms.cern.ch/crab3/crab.sh
   ```
   
   and source it

4. Override SCRAM_ARCH to trick scram
   
   ```bash
   SCRAM_ARCH=slc7_amd64_gcc700
   ```

5. Create a folder ~/CMSSW and
   
   ```bash
   mkdir -p ~/CMSSW
   cd ~/CMSSW
   cmsrel CMSSW_10_6_29
   ```

6. Add at the end of .zshrc (run cmssw only when you need it):
   
   **ROOT DON'T WORK UNDER CMSSW**
   
   ```bash
   cmssw() {
     cd ~/CMSSW/CMSSW_10_6_29/src
     cmsenv
     cd ~
   }
   
   ```

7. To get root working:
   
   - Install from AUR ncurses5-compat-libs
   
   - Run root with(create an alias): LD_PRELOAD=/usr/lib/libfreetype.so root

## Configure the grid

1. Install voms-client:
- Download the last release with wget [Releases · italiangrid/voms-clients · GitHub](https://github.com/italiangrid/voms-clients/releases)

- Unpack and build with
  
  ```bash
  #install maven and java first
  mvn package
  ```
  
  and move the binaries (inside a tarball) in you $PATH
2. Copy all grid certificates from lxplus
   
   ```bash
   sudo mkdir -p /etc/grid-security
   cd /etc
   sudo rsync -avAXEWSlHh pviscone@lxplus.cern.ch:/etc/grid-security . --no-compress --info=progress2
   ```

3. Copy personal certificates form gridui (NOT LXPLUS, for openssh reasons)
   
   ```bash
   cd
   mkdir .globus
   rsync -avAXEWSlHh viscone@gridui.pi.infn.it:~/.globus . --no-compress --info=progress2
   ```

4. Copy vomses locations
   
   ```bash
   cd /etc
   sudo rsync -avAXEWSlHh pviscone@lxplus.cern.ch:/etc/vomses . --no-compress --info=progress2
   ```

## Configure EOS

Use the cernbox application

## Set some alias

in .zshrc

```bash
export TIER="T2_IT_Pisa"
export xrd="root://cms-xrd-global.cern.ch/"
export lxplus="pviscone@lxplus.cern.ch"
export gridui="viscone@gridui.pi.infn.it"
export galil="viscone@galilinux.pi.infn.it"

alias sshlxplus="ssh -Y pviscone@lxplus.cern.ch"
alias sshgridui="ssh -Y viscone@gridui.pi.infn.it"
alias sshgalil="ssh -Y viscone@galilinux.pi.infn.it"
alias voms="voms-proxy-init --rfc --voms cms -valid 192:00"

cmssw() {
  cd ~/CMSSW/CMSSW_10_6_29/src
  cmsenv
  cd ~
}


scp(){
  rsync -avAXEWSlHh "$@" --no-compress --info=progress2
}

```

# TODO

- set the alias with screen 
