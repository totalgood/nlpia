## References

* best blog post/tutorial: https://tech.amikelive.com/node-669/guide-installing-cuda-toolkit-9-1-on-ubuntu-16-04/
* official docs stink: https://developer.nvidia.com/c
* download cuda-toolkit: https://developer.nvidia.com/cuda-downloads (chose deb-network version) 

1. uninstall all cuda and nvidia drivers with 

```bash
sudo apt uninstall nvidia*

# uninstalls these:
# libcuda1-381* libcuinj64-7.5* nvidia-381* nvidia-cuda-doc* nvidia-cuda-gdb*
# nvidia-cuda-toolkit* nvidia-opencl-dev*
# nvidia-opencl-icd-381* nvidia-prime* nvidia-profiler* nvidia-settings*
# nvidia-visual-profiler*
```

2. Prepare build tools

```
sudo apt-get install linux-headers-$(uname -r)
```

2. install the nvidia drivers compatible with the latest cuda-toolkit (9.2)

3. install full anaconda for python 3.6 then

```bash
chown -R hobs:root /opt
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
chmod +x Anaconda3-5.2.0-Linux-x86_64.sh
./Anaconda3-5.2.0-Linux-x86_64.sh
# when asked about install path chose /opt/anaconda
```

4. install keras

```bash
conda install conda
conda install pip
pip install --upgrade pip
conda install tensorflow-gpu
conda install keras-gpu
```


