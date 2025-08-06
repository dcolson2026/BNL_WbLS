_SULI 2025 internship scripts and documentation_

Starting with processing the real data:
  the raw data from the root files is not only raw but also incorrect. timing corrections must be made to account for daisy chain and channel delays. this is handled by
  my_analysis/format_for_NN_more_tagging.py where the user can change various things depending on how/what they want to process the data.
  First, source /media/disk_o/cluster_match/bin/activate. This is just a python venv, but it's important that the data gets pickled in a numpy version compatible with the NN. If
  it is being really finicky, here is the list of all the packages I used that worked. Most importantly was the numpy version
  
  Package            Version
------------------ -----------
asttokens          3.0.0
awkward            2.8.4
awkward_cpp        46
black              25.1.0
click              8.1.8
comm               0.2.2
cramjam            2.10.0
debugpy            1.8.14
decorator          5.2.1
exceptiongroup     1.3.0
executing          2.2.0
fsspec             2025.5.1
importlib_metadata 8.7.0
ipykernel          6.29.5
ipython            8.18.1
jedi               0.19.2
jupyter_client     8.6.3
jupyter_core       5.8.1
matplotlib-inline  0.1.7
mypy_extensions    1.1.0
nest-asyncio       1.6.0
numpy              1.24.4
packaging          25.0
parso              0.8.4
pathspec           0.12.1
pexpect            4.9.0
pip                23.0.1
platformdirs       4.3.8
prompt_toolkit     3.0.51
psutil             7.0.0
ptyprocess         0.7.0
pure_eval          0.2.3
Pygments           2.19.2
python-dateutil    2.9.0.post0
pyzmq              27.0.0
setuptools         58.1.0
six                1.17.0
stack-data         0.6.3
tomli              2.2.1
tornado            6.5.1
traitlets          5.14.3
typing_extensions  4.14.0
uproot             5.6.2
wcwidth            0.2.13
xxhash             3.5.0
zipp               3.23.0

  Secondly, you can change the phase of data you want to analyze. simply change the file path.
  Thirdly, if you are looking at water data, add in the if statement that checks for files that contain "water". This makes sure that injection files and extraneous ones are
  excluded
  Fourth, change the peak method. Currently, the peak of the PMT pulse can be found using CFD or weighted_avg. This is due to the 500MHz sampling rate being too coarse;
  therefore, we need to do some kind of interpolation. There is lots of room for improvement here, perhaps by making a CNN trained on lognormal PMT pulses
  Fifth, ensure that the CFD fraction is accurate with respect to the training data. 
  Sixth, of course change the output pickle file name with some numbering/naming scheme that is useful to you. for exmaple, I might do 07_29_25_disk_d_phase_3_weight to get
  the data, disk, phase, and peak finding method. In a separate paper notebook I keep track of what was tested on certain dates, which is particularly important since you
  can change what kind of tagging to focus on.
  Seventh, are corrected root files needed? During a certain time range, the sampling rate for board 5 was mismatched from boards 1-4. This requires an event correction to be
  toggled. Be sure that the corrected root files are put in a designated directory (then you'll only have to correct them once)
  Eighth and most importantly, at line 793 you can change what kind of tagging you are looking at. Previously, I was filtering for events where TP 3,4, or 5 were hit AND at least
  one BP hit.
  Lastly, scp the data over to whatever cluster you are doing the event reconstruction on.
  This outputs the all-sensor data, per-sensor data, which make more sense after reading the instructions below:

Env for neural networks:
  before i describe how the NNs work, you need the proper env. The conda env was extremely painful to set up. These were the packages that made everything work without error
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                       2_gnu    conda-forge
absl-py                   2.3.0                    pypi_0    pypi
asttokens                 3.0.0                    pypi_0    pypi
astunparse                1.6.3                    pypi_0    pypi
awkward                   2.8.4                    pypi_0    pypi
awkward-cpp               46                       pypi_0    pypi
bzip2                     1.0.8                h4bc722e_7    conda-forge
ca-certificates           2025.6.15            hbd8a1cb_0    conda-forge
cachetools                5.5.2                    pypi_0    pypi
certifi                   2025.6.15                pypi_0    pypi
charset-normalizer        3.4.2                    pypi_0    pypi
comm                      0.2.2                    pypi_0    pypi
contourpy                 1.3.2                    pypi_0    pypi
cramjam                   2.10.0                   pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
debugpy                   1.8.14                   pypi_0    pypi
decorator                 5.2.1                    pypi_0    pypi
exceptiongroup            1.3.0                    pypi_0    pypi
executing                 2.2.0                    pypi_0    pypi
flatbuffers               25.2.10                  pypi_0    pypi
fonttools                 4.58.4                   pypi_0    pypi
fsspec                    2025.5.1                 pypi_0    pypi
gast                      0.4.0                    pypi_0    pypi
google-auth               2.40.3                   pypi_0    pypi
google-auth-oauthlib      0.4.6                    pypi_0    pypi
google-pasta              0.2.0                    pypi_0    pypi
grpcio                    1.73.0                   pypi_0    pypi
h5py                      3.14.0                   pypi_0    pypi
hitman                    0.2                      pypi_0    pypi
idna                      3.10                     pypi_0    pypi
importlib-metadata        8.7.0                    pypi_0    pypi
ipykernel                 6.29.5                   pypi_0    pypi
ipython                   8.37.0                   pypi_0    pypi
jedi                      0.19.2                   pypi_0    pypi
jupyter-client            8.6.3                    pypi_0    pypi
jupyter-core              5.8.1                    pypi_0    pypi
keras                     2.11.0                   pypi_0    pypi
kiwisolver                1.4.8                    pypi_0    pypi
ld_impl_linux-64          2.43                 h1423503_5    conda-forge
libclang                  18.1.1                   pypi_0    pypi
libexpat                  2.7.0                h5888daf_0    conda-forge
libffi                    3.4.6                h2dba641_1    conda-forge
libgcc                    15.1.0               h767d61c_3    conda-forge
libgcc-ng                 15.1.0               h69a702a_3    conda-forge
libgomp                   15.1.0               h767d61c_3    conda-forge
liblzma                   5.8.1                hb9d3cd8_2    conda-forge
libnsl                    2.0.1                hb9d3cd8_1    conda-forge
libsqlite                 3.50.1               h6cd9bfd_6    conda-forge
libuuid                   2.38.1               h0b41bf4_0    conda-forge
libxcrypt                 4.4.36               hd590300_1    conda-forge
libzlib                   1.3.1                hb9d3cd8_2    conda-forge
markdown                  3.8.2                    pypi_0    pypi
markupsafe                3.0.2                    pypi_0    pypi
matplotlib                3.10.3                   pypi_0    pypi
matplotlib-inline         0.1.7                    pypi_0    pypi
ncurses                   6.5                  h2d0b736_3    conda-forge
nest-asyncio              1.6.0                    pypi_0    pypi
numpy                     1.24.4                   pypi_0    pypi
oauthlib                  3.3.1                    pypi_0    pypi
openssl                   3.5.0                h7b32b05_1    conda-forge
opt-einsum                3.4.0                    pypi_0    pypi
packaging                 25.0                     pypi_0    pypi
parso                     0.8.4                    pypi_0    pypi
pexpect                   4.9.0                    pypi_0    pypi
pillow                    11.2.1                   pypi_0    pypi
pip                       25.1.1             pyh8b19718_0    conda-forge
platformdirs              4.3.8                    pypi_0    pypi
prompt-toolkit            3.0.51                   pypi_0    pypi
protobuf                  3.19.6                   pypi_0    pypi
psutil                    7.0.0                    pypi_0    pypi
ptyprocess                0.7.0                    pypi_0    pypi
pure-eval                 0.2.3                    pypi_0    pypi
pyasn1                    0.6.1                    pypi_0    pypi
pyasn1-modules            0.4.2                    pypi_0    pypi
pygments                  2.19.2                   pypi_0    pypi
pyparsing                 3.2.3                    pypi_0    pypi
python                    3.10.18         hd6af730_0_cpython    conda-forge
python-dateutil           2.9.0.post0              pypi_0    pypi
pyzmq                     27.0.0                   pypi_0    pypi
readline                  8.2                  h8c095d6_2    conda-forge
requests                  2.32.4                   pypi_0    pypi
requests-oauthlib         2.0.0                    pypi_0    pypi
rsa                       4.9.1                    pypi_0    pypi
scipy                     1.15.3                   pypi_0    pypi
setuptools                80.9.0             pyhff2d567_0    conda-forge
six                       1.17.0                   pypi_0    pypi
stack-data                0.6.3                    pypi_0    pypi
tensorboard               2.11.2                   pypi_0    pypi
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1                    pypi_0    pypi
tensorflow                2.11.0                   pypi_0    pypi
tensorflow-estimator      2.11.0                   pypi_0    pypi
tensorflow-io-gcs-filesystem 0.37.1                   pypi_0    pypi
termcolor                 3.1.0                    pypi_0    pypi
tk                        8.6.13          noxft_hd72426e_102    conda-forge
tornado                   6.5.1                    pypi_0    pypi
tqdm                      4.67.1                   pypi_0    pypi
traitlets                 5.14.3                   pypi_0    pypi
typing-extensions         4.14.0                   pypi_0    pypi
tzdata                    2025b                h78e105d_0    conda-forge
uproot                    5.6.2                    pypi_0    pypi
urllib3                   2.5.0                    pypi_0    pypi
wcwidth                   0.2.13                   pypi_0    pypi
werkzeug                  3.1.3                    pypi_0    pypi
wheel                     0.45.1             pyhd8ed1ab_1    conda-forge
wrapt                     1.17.2                   pypi_0    pypi
xxhash                    3.5.0                    pypi_0    pypi
zipp                      3.23.0                   pypi_0    pypi

Training the neural networks:
  simulate events using ratpac and GEANT4. There were 8 networks used in my analysis, because each network had 1 attribute from the following 3 types-
    edge or weight: this means either a constant fraction discriminator or weighted averaging was done to find the peaks hit time from the raw data
    numCher or eloss: this describes how the light yield was simulated using rat-pac
    all or per: this describes whether the input charge data was from the PMTs collectively or individually

Testing the neural networks:
  My preprocessing handles the all and per, you just have to change the hit time method. In other words, with just 2 files with different hit time methods, you can test
  all 8 networks. numCher has typically performed the best, but more analysis is included in my report. Use the slurm files included in this repo and run sbatch XslurmfileX.
  The bulk of the work on NN was done by Adam Baldoni and his group at PSU, so consult him with any further questions.

Overall, this is the workflow. Process the raw data to time synchronize it and copy it to NN cluster. Then train the NN and test them on the corresponding processed data.
