_SULI 2025 internship scripts and documentation for 1T_

(look at this raw to see indentation and spacing)

Starting with processing the real data:
  the raw data from the root files is not only raw but also incorrect. timing corrections must be made to account for daisy chain and channel delays. this is handled by
  my_analysis/format_for_NN_more_tagging.py where the user can change various things depending on how/what they want to process the data.
  First, source /media/disk_o/cluster_match/bin/activate. This is just a python venv, but it's important that the data gets pickled in a numpy version compatible with the NN. If
  it is being really finicky, here is the list of all the packages I used that worked. Most importantly was the numpy version. See mvdpro_requirements.txt

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
  see psu_requirements.txt

Training the neural networks:
  simulate events using ratpac and GEANT4. There were 8 networks used in my analysis, because each network had 1 attribute from the following 3 types-
    edge or weight: this means either a constant fraction discriminator or weighted averaging was done to find the peaks hit time from the raw data
    numCher or eloss: this describes how the light yield was simulated using rat-pac
    all or per: this describes whether the input charge data was from the PMTs collectively or individually

Testing the neural networks:
  My preprocessing handles the all and per, you just have to change the hit time method. In other words, with just 2 files with different hit time methods, you can test
  all 8 networks. numCher has typically performed the best, but more analysis is included in my report. Use the slurm files included in this repo and run sbatch XslurmfileX.
  The bulk of the work on NN was done by Adam Baldoni and his group at PSU, so consult him with any further questions.

Plotting NN information:
  These I also wrote entirely and are somewhat explanatory in the titles. 
    entry_exit_plane_plots gives the 2D histograms of reconstructed hits. 
    hit_time_ratio was an attempt to see any difference between prompt Cher and delayed scint, though it does not seem that our detector is sensitive enough yet
    per_sensor_detector_plot gives a nice visual of reconstructions
    real_data_analysis is for analyzing the real data. just paste in the file paths of the reconstruction and run all. Change anything that you are looking at as well
    real_data_analysis_more_tagging used for TP 3,4,5 and hodo data
    simulated_delta_r_and_theta is the same as real_data_analysis but for simulated data

Overall, this is the workflow. Process the raw data to time synchronize it and copy it to NN cluster. Then train the NN and test them on the corresponding processed data. Then
use the psu python notebooks to analyze.
