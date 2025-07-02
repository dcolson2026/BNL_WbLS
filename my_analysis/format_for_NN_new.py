import os
import pickle
import numpy as np
import uproot
import time

# Note that these go in order from bottom, side, supplemental side
# from tof_alpha_source import *
# PMT_x_locations = bottom_PMTs_x + side_PMTs_x + supp_side_PMTs_x
# PMT_y_locations = bottom_PMTs_y + side_PMTs_y + supp_side_PMTs_y
# PMT_z_locations = bottom_PMTs_z + side_PMTs_z + supp_side_PMTs_z
# PMT_keys = bottom_PMT_list + side_PMT_list + supp_side_PMT_list
# PMT_location_dict = {
#     PMT_keys[i]: (PMT_x_locations[i], PMT_y_locations[i], PMT_z_locations[i])
#     for i in range(len(PMT_keys))
# }
IRRELEVANT_CHANNELS = [
    "adc_b1_ch0",
    "adc_b2_ch15",
    "adc_b4_ch12",
    "adc_b4_ch13",
    "adc_b4_ch14",
    "adc_b4_ch15",
]
PMT_location_dict = {
    "adc_b1_ch1": (381.0, -171.45, -677.1),
    "adc_b1_ch2": (381.0, -57.15, -677.1),
    "adc_b1_ch3": (381.0, 57.15, -677.1),
    "adc_b1_ch4": (381.0, 171.45, -677.1),
    "adc_b1_ch5": (190.5, -342.9, -677.1),
    "adc_b1_ch6": (190.5, -228.6, -677.1),
    "adc_b1_ch7": (190.5, -114.3, -677.1),
    "adc_b1_ch8": (190.5, 0.0, -677.1),
    "adc_b1_ch9": (190.5, 114.3, -677.1),
    "adc_b1_ch10": (190.5, 228.6, -677.1),
    "adc_b1_ch11": (190.5, 342.9, -677.1),
    "adc_b1_ch12": (0.0, -400.05, -677.1),
    "adc_b1_ch13": (0.0, -285.75, -677.1),
    "adc_b1_ch14": (0.0, -171.45, -677.1),
    "adc_b1_ch15": (0.0, -57.15, -677.1),
    "adc_b2_ch0": (0.0, 57.15, -677.1),
    "adc_b2_ch1": (0.0, 171.45, -677.1),
    "adc_b2_ch2": (0.0, 285.75, -677.1),
    "adc_b2_ch3": (0.0, 400.05, -677.1),
    "adc_b2_ch4": (-190.5, -342.9, -677.1),
    "adc_b2_ch5": (-190.5, -228.6, -677.1),
    "adc_b2_ch6": (-190.5, -114.3, -677.1),
    "adc_b2_ch7": (-190.5, 0.0, -677.1),
    "adc_b2_ch8": (-190.5, 114.3, -677.1),
    "adc_b2_ch9": (-190.5, 228.6, -677.1),
    "adc_b2_ch10": (-190.5, 342.9, -677.1),
    "adc_b2_ch11": (-381.0, -171.45, -677.1),
    "adc_b2_ch12": (-381.0, -57.15, -677.1),
    "adc_b2_ch13": (-381.0, 57.15, -677.1),
    "adc_b2_ch14": (-381.0, 171.45, -677.1),
    "adc_b3_ch0": (-532.955, 0.0, -495.3),
    "adc_b3_ch1": (-532.955, 0.0, -336.55),
    "adc_b3_ch2": (-532.955, 0.0, 222.25),
    "adc_b3_ch3": (-532.955, 0.0, 393.7),
    "adc_b3_ch4": (532.955, 0.0, -495.3),
    "adc_b3_ch5": (532.955, 0.0, -336.55),
    "adc_b3_ch6": (532.955, 0.0, 222.25),
    "adc_b3_ch7": (532.955, 0.0, 393.7),
    "adc_b3_ch8": (0.0, -532.955, -495.3),
    "adc_b3_ch9": (0.0, -532.955, -336.55),
    "adc_b3_ch10": (0.0, -532.955, 222.25),
    "adc_b3_ch11": (0.0, -532.955, 393.7),
    "adc_b3_ch12": (0.0, 532.955, -495.3),
    "adc_b3_ch13": (0.0, 532.955, -336.55),
    "adc_b3_ch14": (0.0, 532.955, 222.25),
    "adc_b3_ch15": (0.0, 532.955, 393.7),
    "adc_b4_ch0": (-376.8561, -376.8561, -211.0232),
    "adc_b4_ch1": (-376.8561, -376.8561, -41.1607),
    "adc_b4_ch2": (-376.8561, -376.8561, 128.7018),
    "adc_b4_ch3": (376.8561, 376.8561, -211.0232),
    "adc_b4_ch4": (376.8561, 376.8561, -41.1607),
    "adc_b4_ch5": (376.8561, 376.8561, 128.7018),
    "adc_b4_ch6": (376.8561, -376.8561, -211.0232),
    "adc_b4_ch7": (376.8561, -376.8561, -41.1607),
    "adc_b4_ch8": (376.8561, -376.8561, 128.7018),
    "adc_b4_ch9": (-376.8561, 376.8561, -211.0232),
    "adc_b4_ch10": (-376.8561, 376.8561, -41.1607),
    "adc_b4_ch11": (-376.8561, 376.8561, 128.7018),
}

# temporarily using a low statistics one here
PMT_channel_delay_dict = {
    "adc_b1_ch1": np.float64(7.8599166820699455),
    "adc_b1_ch2": np.float64(4.2279287488665505),
    "adc_b1_ch3": np.float64(5.574034508625212),
    "adc_b1_ch4": np.float64(1.462657718896561),
    "adc_b1_ch5": np.float64(5.46018892397303),
    "adc_b1_ch6": np.float64(5.409302133389168),
    "adc_b1_ch7": np.float64(3.9025018950943133),
    "adc_b1_ch8": np.float64(3.46381735976686),
    "adc_b1_ch9": np.float64(3.885940297701321),
    "adc_b1_ch10": np.float64(4.053950701931467),
    "adc_b1_ch11": np.float64(5.574917569091319),
    "adc_b1_ch12": np.float64(3.4946922002103027),
    "adc_b1_ch13": np.float64(2.660567554458799),
    "adc_b1_ch14": np.float64(3.9224651796362338),
    "adc_b1_ch15": np.float64(4.611150134965591),
    "adc_b2_ch0": np.float64(5.657053247743757),
    "adc_b2_ch1": np.float64(3.9912637690502324),
    "adc_b2_ch2": np.float64(4.004392700853049),
    "adc_b2_ch3": np.float64(3.5197659369335343),
    "adc_b2_ch4": np.float64(2.4646066520412377),
    "adc_b2_ch5": np.float64(3.59695668842358),
    "adc_b2_ch6": np.float64(6.1581988684208735),
    "adc_b2_ch7": np.float64(4.4218622173912525),
    "adc_b2_ch8": np.float64(0.6521201336279718),
    "adc_b2_ch9": np.float64(4.226312224499658),
    "adc_b2_ch10": np.float64(5.533503195204535),
    "adc_b2_ch11": np.float64(5.29138682467591),
    "adc_b2_ch12": np.float64(4.0094185548337125),
    "adc_b2_ch13": np.float64(4.08831536665447),
    "adc_b2_ch14": np.float64(2.5962437226528117),
    "adc_b3_ch0": np.float64(3.442589791305678),
    "adc_b3_ch1": np.float64(1.6667315660917326),
    "adc_b3_ch2": np.float64(3.423351974027864),
    "adc_b3_ch3": np.float64(6.209936881382594),
    "adc_b3_ch4": np.float64(1.3518561473109876),
    "adc_b3_ch5": np.float64(1.8722080052467223),
    "adc_b3_ch6": np.float64(4.0035516253739),
    "adc_b3_ch7": np.float64(3.2043085739745996),
    "adc_b3_ch8": np.float64(3.8311094721486803),
    "adc_b3_ch9": np.float64(4.5314879480281345),
    "adc_b3_ch10": np.float64(2.434797802550851),
    "adc_b3_ch11": np.float64(3.767981411743301),
    "adc_b3_ch12": np.float64(2.396616695331079),
    "adc_b3_ch13": np.float64(3.4128794605255703),
    "adc_b3_ch14": np.float64(3.246248631680017),
    "adc_b3_ch15": np.float64(1.4297591545820207),
    "adc_b4_ch0": np.float64(2.704113081726384),
    "adc_b4_ch1": np.float64(2.0354747883796236),
    "adc_b4_ch2": np.float64(4.41186642937317),
    "adc_b4_ch3": np.float64(3.7450287508165627),
    "adc_b4_ch4": np.float64(2.039541912010301),
    "adc_b4_ch5": np.float64(1.9065206786959155),
    "adc_b4_ch6": np.float64(2.7356183903993316),
    "adc_b4_ch7": np.float64(3.821243571830856),
    "adc_b4_ch8": np.float64(10.311279529931609),
    "adc_b4_ch9": np.float64(2.8341494076225557),
    "adc_b4_ch10": np.float64(4.047511889036665),
    "adc_b4_ch11": np.float64(4.251758108079468),
}

def get_1t_info(fname: str):
    f = uproot.open(fname)
    daq = f["daq"]
    # sometimes this isn't in the root file
    if "run_info" in f:
        run_info = f["run_info"]
    else:
        run_info = None
    daqkeys = daq.keys()
    traces = {}
    for key in daq.keys():
        if "adc" in key:
            traces[key] = daq[key].array(library="np")
    event_ttt1 = daq["event_ttt_1"].array(library="np")
    event_ttt2 = daq["event_ttt_2"].array(library="np")
    event_ttt3 = daq["event_ttt_3"].array(library="np")
    event_ttt4 = daq["event_ttt_4"].array(library="np")
    event_ttt5 = daq["event_ttt_5"].array(library="np")
    event_id = daq["event_id"].array(library="np")
    event_sanity = daq["event_sanity"].array(library="np")
    # event_ttt = daq['event_ttt'].array(library='np')

    return (
        traces,
        event_ttt1.astype(np.int64),
        event_ttt2.astype(np.int64),
        event_ttt3.astype(np.int64),
        event_ttt4.astype(np.int64),
        event_ttt5.astype(np.int64),
        event_id,
        event_sanity,
        daqkeys,
        run_info,
    )

def base_and_flip(waveform):
    """Subtract baseline and reflect over x axis"""
    positive_waveform = (waveform - np.median(waveform)) * (-1)
    return positive_waveform

def weighted_average_hit_time(waveform, window_size=10):
    """Do weighted average in window around pulse. Returns float value at which
    hit time occurred"""
    # Call correction algorithm
    waveform = base_and_flip(waveform)

    # Make into list
    waveform = list(waveform)
    
    # Find index of max (the pulse peak)
    peak_index = waveform.index(max(waveform))
    
    # Define window bounds
    half_window = window_size // 2
    start = max(0, peak_index - half_window)
    end = min(len(waveform), peak_index + half_window + 1)
    
    # Get time (index) and amplitude (value) in the window
    times = list(range(start, end))
    amplitudes = waveform[start:end]
    
    # Compute weighted average hit time
    numerator = sum(t * a for t, a in zip(times, amplitudes))
    denominator = sum(amplitudes)
    
    if denominator == 0:
        return None  # Avoid divide-by-zero
    return numerator / denominator

def get_channel_charge(waveform):
    """Takes in a raw waveform. Does baseline subtraction, makes it positive, make window of
    size 60ns / 30 sample, integrate by just taking sum (nothing fancy), divide by 50 (resistance),
    returns charge in pC"""
    based_flipped = base_and_flip(waveform)
    time_of_max = np.argmax(based_flipped)
    charge_pC = np.sum(based_flipped[time_of_max - 5: time_of_max + 5]) / 50
    return charge_pC

def waveform_daisy_correction(waveform, boardID):
    if (boardID < 1) or (boardID > 4):
        print("Bad BoardID")
        return False
    elif boardID != 1:
        return waveform[24 * (4 - boardID) : -24 * (boardID - 1)]
    else:
        return waveform[24 * 3 :]

def is_pulse(waveform: np.ndarray, range_min: int = 0, range_max: int = 1928) -> bool:
    """Takes in a daisy corrected waveform and looks in a given range to see if there is a pulse.
    For example, you can use some range around an alpha PMT hit if looking for just alpha detections.
    """
    threshold_sigma = 15
    # this is arbitrary, and hopefully sufficient
    baseline = np.median(waveform[:int(0.5 * len(waveform))])
    noise_std = np.std(waveform[:int(0.5 * len(waveform))])
    deviation = np.abs(waveform - baseline)
    threshold = threshold_sigma * noise_std

    # Get all indices where deviation exceeds threshold
    pulse_indices = np.where(deviation > threshold)[0]
    for i in pulse_indices:
        if range_min <= i <= range_max:
            return True
    return False

def b1_ch0_or_b2_ch15_detections(traces):
    """These are bottom paddle channels, gets list of events with detections."""
    b1_ch0_only_list = []
    b2_ch15_only_list = []

    b1_ch0_waveforms_list = traces["adc_b1_ch0"]
    b2_ch15_waveforms_list = traces["adc_b2_ch15"]
    for i, waveform in enumerate(b1_ch0_waveforms_list):
        if is_pulse(waveform):  # this is arbitrary, and hopefully this is sufficient
            b1_ch0_only_list.append(i)
    for i, waveform in enumerate(b2_ch15_waveforms_list):
        if is_pulse(waveform):  # this is arbitrary, and hopefully this is sufficient
            b2_ch15_only_list.append(i)
    return b1_ch0_only_list, b2_ch15_only_list

def b4_ch13_or_ch14_detections(traces):
    """These are top paddle channels, gets list of events with detections."""
    b4_ch13_or_ch14_detection_list = []
    waveforms_list = traces["adc_b4_ch13"] + traces["adc_b4_ch14"]
    for i, waveform in enumerate(waveforms_list):
        if is_pulse(waveform):  # this is arbitrary, and hopefully this is sufficient
            b4_ch13_or_ch14_detection_list.append(i)
    return b4_ch13_or_ch14_detection_list

def top_paddle_event_list(traces) -> list[int]:
    """Returns a list of event indices that correspond to top_paddle trigger events.
    This means b4_ch13 OR b4_ch14 has a signal AND that the superposition of signals
    for that event lies in the time range for top_paddle events."""

    twice_checked_top_paddle_event_index_list = []
    num_events = len(traces["adc_b2_ch1"])  # pick arbitrary PMT, all same length
    top_paddle_PMT_events = b4_ch13_or_ch14_detections(traces)  # fix for top_paddle

    for i in range(num_events):
        corrected_waveforms_per_event = []
        # waveform loop to get the i_th waveform for each PMT
        for key in traces.keys():
            if ("b5" in key) or (key in IRRELEVANT_CHANNELS):  # Adam said disregard
                continue
            board_num = int(key[5])
            uncorrected_waveform = traces[key][i]
            
            corrected_waveforms_per_event.append(
                waveform_daisy_correction(uncorrected_waveform, board_num)
            )
        summed_waveform = np.sum(corrected_waveforms_per_event, axis=0)
        peak_sample_time_ns = np.argmin(summed_waveform) * 2  # converts to ns
        # rough estimate of time range
        if 750 < peak_sample_time_ns and i in top_paddle_PMT_events:
            twice_checked_top_paddle_event_index_list.append(i)
    return twice_checked_top_paddle_event_index_list

def get_all_sensor_input(fname: str, peak_method: str):
    """Takes in a file path and a peak_method, which is either CFD or weighted avg.
    Then the information for all-sensor chargenet AND all-sensor hitnet is returned."""

    all_events = []
    bottom_paddle_tags_list = []
    
    traces = get_1t_info(fname)[0]
    top_paddle_events = top_paddle_event_list(traces)
    bottom_paddle_1, bottom_paddle_2 = b1_ch0_or_b2_ch15_detections(traces)
    print("BP detections:",bottom_paddle_1,bottom_paddle_2)

    for i in top_paddle_events:
        #print("starting new TP event")
        hitnet_input = [[], [], [], [], []]
        chargenet_input = []
        sum_of_charges_of_all_hits = 0
        num_of_hits = 0

        # Could have just gotten this intersection I guess
        bottom_paddle_event_tag = 0
        if (i in bottom_paddle_1) and (i in bottom_paddle_2):
            bottom_paddle_event_tag = 3
        elif i in bottom_paddle_1:
            bottom_paddle_event_tag = 1
        elif i in bottom_paddle_2:
            bottom_paddle_event_tag = 2
        if bottom_paddle_event_tag == 0: # skip, can't do true comparison
            continue

        # waveform loop to get the i_th waveform for each PMT
        for key in traces.keys():
            if ("b5" in key) or (key in IRRELEVANT_CHANNELS):  # disregard
                continue
            uncorrected_waveform = traces[key][i]
            board_num = int(key[5])
            
            # perform daisy correction and change to ns (500MHz sampling)
            daisy_corrected_waveform = waveform_daisy_correction(
                uncorrected_waveform, board_num
            ) * 2

            # skip if not a pulse (add in pulse window?)
            # GET ONLY THE NECESSARY EVENTS from TP trigger!!!
            if not is_pulse(daisy_corrected_waveform, range_min=750):
                continue

            # factors in the channel delay and reassigns it to daisy_corrected_waveform
            daisy_corrected_waveform = [x - PMT_channel_delay_dict[key] for x in daisy_corrected_waveform]

            # Fix wonky waveforms from Oct 31, 2024 by ensuring charge > 0
            # potentially not a rigorous fix
            waveform_charge = get_channel_charge(daisy_corrected_waveform)
            if waveform_charge < 0:
                continue

            print("hit for", key)
            # hitnet input as we go
            hitnet_input[0].append(PMT_location_dict[key][0])
            hitnet_input[1].append(PMT_location_dict[key][1])
            hitnet_input[2].append(PMT_location_dict[key][2])
            if peak_method == "CFD":
                hitnet_input[3].append(-1)
            elif peak_method == "W_avg":
                hitnet_input[3].append(weighted_average_hit_time(daisy_corrected_waveform))
            hitnet_input[4].append(1)

            # chargenet values to later input
            sum_of_charges_of_all_hits += waveform_charge
            num_of_hits += 1

        # chargenet input
        chargenet_input.append(sum_of_charges_of_all_hits)
        chargenet_input.append(num_of_hits)

        event = {
            "hits": np.stack(hitnet_input, axis=1),
            "total_charge": np.stack(chargenet_input),
        }

        # only append if there's anything to reco
        if num_of_hits > 0:
            all_events.append(event)
            bottom_paddle_tags_list.append(bottom_paddle_event_tag)
            print("hitnet and chargent input")
            print(hitnet_input, "\n", chargenet_input)
            print("bp tag", bottom_paddle_event_tag)
    return all_events, bottom_paddle_tags_list


# num_files = -1
# phase_directory = "/media/disk_d/WbLS-DATA/raw_root/phase3/muon/" # Oct 31, 2024
phase_directory = "/media/disk_e/WbLS-DATA/raw_root/phase3/muon/" # Nov 13, 2024
# phase_directory = "/media/disk_a/WbLS-DATA/raw_root/phase6/muon/" # Jan 07, 2025
# phase_directory = "/media/disk_b/WbLS-DATA/raw_root/phase6/muon/" # Dec 19, 2024
# phase_directory = "/media/disk_e/WbLS-DATA/raw_root/phase4/muon/" # Dec 03, 2024
# phase_directory = "/media/disk_k/WbLS-DATA/raw_root/phase8/muon/" # Mar 11, 2025
file_paths_for_ch_delays = [phase_directory + str(f) for f in os.listdir(phase_directory) if os.path.isfile(os.path.join(phase_directory, f))]

all_events_for_phase = []
all_bottom_paddle_tags_for_phase = []
for fileee in file_paths_for_ch_delays:#[:num_files]:
    if "water" not in fileee: continue # network was trained on water
    print("starting new file", fileee)
    try:
        all_eventsy, bottom_paddle_tags_listy = get_all_sensor_input(fileee, "W_avg")
        all_events_for_phase.extend(all_eventsy)
        all_bottom_paddle_tags_for_phase.extend(bottom_paddle_tags_listy)
    except Exception as e:
        print(f"Skipped {fileee} due to error: {e}")
        time.sleep(3)

data_to_save = {
    "all_events_for_phase": all_events_for_phase,
    "all_bottom_paddle_tags_for_phase": all_bottom_paddle_tags_for_phase
}

print("completed:", len(all_events_for_phase), len(all_bottom_paddle_tags_for_phase))
with open(f'/media/disk_o/my_pickles/processed_data_for_disk_e_phase_3_all.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

# scp /path/to/local/file username@cluster.server.edu:/path/to/cluster/destination/
# scp /path/to/local/file username@cluster.server.edu:/path/to/cluster/destination/
# scp /media/disk_o/my_pickles/processed_data_for_disk_k_phase_8_30.pkl dzc5938@submit.hpc.psu.edu:/storage/group/dfc13/default/dcolson/my_pickles
