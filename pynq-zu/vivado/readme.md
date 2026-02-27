# Installing Vivado 2022.2 on Ubuntu 24.04.3 LTS
The installer for Vivado 2022.2 does not include the necessary dependencies that it needs to complete the installation/run the software. This will likely result in the computer freezing in the last step of the installation, not being able to complete.

To resolve this issue, the dependencies `libtinfo5` and `libncurses5` need to be installed **prior** to running the installer for Vivado 2022.2. To do this, download the script `vivado_dep_2022-2.sh` from this repo. Open a terminal in the directory with the script and run the following:
```
chmod +x ./vivado_dep_2022-2.sh
sudo ./vivado_dep_2022-2.sh
```
These commands will first allow the shell script to be ran as an executable, then provide it with sudo permissions. It will automatically grab the correct version of the dependencies as required by Vivado, and install them. Please note that the script targets AMD64 based systems (x86).
