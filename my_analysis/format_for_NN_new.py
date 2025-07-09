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

# using ALL data from 4 phases, see /media/disk_o/my_analysis/medians.py
# PMT_channel_delay_dict = {
#     "adc_b1_ch1": np.float64(8.030119938430738),
#     "adc_b1_ch2": np.float64(5.160596843708807),
#     "adc_b1_ch3": np.float64(6.183129915008303),
#     "adc_b1_ch4": np.float64(2.1006828929114034),
#     "adc_b1_ch5": np.float64(6.2045754222843446),
#     "adc_b1_ch6": np.float64(6.161676253769009),
#     "adc_b1_ch7": np.float64(4.561973370592692),
#     "adc_b1_ch8": np.float64(4.059154543523331),
#     "adc_b1_ch9": np.float64(4.527702749808608),
#     "adc_b1_ch10": np.float64(4.89672876798353),
#     "adc_b1_ch11": np.float64(6.264616928546467),
#     "adc_b1_ch12": np.float64(4.0887738673544405),
#     "adc_b1_ch13": np.float64(3.6559960302367407),
#     "adc_b1_ch14": np.float64(4.5739947702690955),
#     "adc_b1_ch15": np.float64(5.575061440615521),
#     "adc_b2_ch0": np.float64(6.316710428600398),
#     "adc_b2_ch1": np.float64(4.6156615340751275),
#     "adc_b2_ch2": np.float64(4.651063059501854),
#     "adc_b2_ch3": np.float64(4.240215238749752),
#     "adc_b2_ch4": np.float64(3.29829818445931),
#     "adc_b2_ch5": np.float64(3.9065740208083692),
#     "adc_b2_ch6": np.float64(6.998608781647993),
#     "adc_b2_ch7": np.float64(5.218269533126902),
#     "adc_b2_ch8": np.float64(1.5691269263153176),
#     "adc_b2_ch9": np.float64(5.048373360049517),
#     "adc_b2_ch10": np.float64(6.101505155220909),
#     "adc_b2_ch11": np.float64(5.9255042831075135),
#     "adc_b2_ch12": np.float64(4.576728227848958),
#     "adc_b2_ch13": np.float64(4.806394551240132),
#     "adc_b2_ch14": np.float64(3.4518309802066334),
#     "adc_b3_ch0": np.float64(4.219983404983012),
#     "adc_b3_ch1": np.float64(2.3410189582944896),
#     "adc_b3_ch2": np.float64(4.38371641420724),
#     "adc_b3_ch3": np.float64(6.840023772731689),
#     "adc_b3_ch4": np.float64(2.1118156288779253),
#     "adc_b3_ch5": np.float64(2.5750972348009955),
#     "adc_b3_ch6": np.float64(4.730938465704396),
#     "adc_b3_ch7": np.float64(3.9063029644947553),
#     "adc_b3_ch8": np.float64(4.337709021825615),
#     "adc_b3_ch9": np.float64(5.3446182004022),
#     "adc_b3_ch10": np.float64(3.2323137947678804),
#     "adc_b3_ch11": np.float64(4.585152794796352),
#     "adc_b3_ch12": np.float64(3.1823386547200014),
#     "adc_b3_ch13": np.float64(4.146583102970336),
#     "adc_b3_ch14": np.float64(4.054654654843673),
#     "adc_b3_ch15": np.float64(2.008745263456964),
#     "adc_b4_ch0": np.float64(3.728169312817592),
#     "adc_b4_ch1": np.float64(3.172271565996006),
#     "adc_b4_ch2": np.float64(5.504018432122031),
#     "adc_b4_ch3": np.float64(4.67892164680153),
#     "adc_b4_ch4": np.float64(2.8893489155116834),
#     "adc_b4_ch5": np.float64(2.8683819880358103),
#     "adc_b4_ch6": np.float64(3.9172990994935835),
#     "adc_b4_ch7": np.float64(4.9451057648992744),
#     "adc_b4_ch8": np.float64(11.154814540200347),
#     "adc_b4_ch9": np.float64(3.349086284111587),
#     "adc_b4_ch10": np.float64(4.758447378646148),
#     "adc_b4_ch11": np.float64(4.971188071773195),
# }
PMT_channel_delay_dict = {
    "adc_b1_ch1": np.float64(1.0211744381089476),
    "adc_b1_ch2": np.float64(3.803536405528817),
    "adc_b1_ch3": np.float64(4.816411066307445),
    "adc_b1_ch4": np.float64(0.6193847551627432),
    "adc_b1_ch5": np.float64(4.635112287600739),
    "adc_b1_ch6": np.float64(4.756969123683583),
    "adc_b1_ch7": np.float64(3.295062163112044),
    "adc_b1_ch8": np.float64(2.7887379301879167),
    "adc_b1_ch9": np.float64(3.6017375458048733),
    "adc_b1_ch10": np.float64(4.209549076904713),
    "adc_b1_ch11": np.float64(5.0754560196043315),
    "adc_b1_ch12": np.float64(2.5681392266603633),
    "adc_b1_ch13": np.float64(2.5458820435829064),
    "adc_b1_ch14": np.float64(3.391507676637081),
    "adc_b1_ch15": np.float64(4.247887581944376),
    "adc_b2_ch0": np.float64(5.0361428107884),
    "adc_b2_ch1": np.float64(3.5275556662494503),
    "adc_b2_ch2": np.float64(3.8343425735251886),
    "adc_b2_ch3": np.float64(3.0551557307539454),
    "adc_b2_ch4": np.float64(1.9356440342982673),
    "adc_b2_ch5": np.float64(2.6212142475072677),
    "adc_b2_ch6": np.float64(5.9220512037766),
    "adc_b2_ch7": np.float64(4.136027879518686),
    "adc_b2_ch8": np.float64(0.7486842013577144),
    "adc_b2_ch9": np.float64(4.055283738274365),
    "adc_b2_ch10": np.float64(4.843713540805526),
    "adc_b2_ch11": np.float64(4.3796015702769715),
    "adc_b2_ch12": np.float64(3.1962951597174873),
    "adc_b2_ch13": np.float64(3.6512165466621984),
    "adc_b2_ch14": np.float64(2.3569460756383833),
    "adc_b3_ch0": np.float64(3.6671043312680047),
    "adc_b3_ch1": np.float64(1.8365771110283582),
    "adc_b3_ch2": np.float64(4.251500439012001),
    "adc_b3_ch3": np.float64(6.913629071292966),
    "adc_b3_ch4": np.float64(1.4004316462576094),
    "adc_b3_ch5": np.float64(1.9891596118248724),
    "adc_b3_ch6": np.float64(4.560482558547185),
    "adc_b3_ch7": np.float64(3.6410986360349145),
    "adc_b3_ch8": np.float64(3.423371201008272),
    "adc_b3_ch9": np.float64(5.157732145850491),
    "adc_b3_ch10": np.float64(2.9996230186799298),
    "adc_b3_ch11": np.float64(3.990100536262517),
    "adc_b3_ch12": np.float64(2.6741284939919145),
    "adc_b3_ch13": np.float64(3.609122818417561),
    "adc_b3_ch14": np.float64(3.9072421230135106),
    "adc_b3_ch15": np.float64(1.3570921414719277),
    "adc_b4_ch0": np.float64(3.5789606695594984),
    "adc_b4_ch1": np.float64(2.971517103641033),
    "adc_b4_ch2": np.float64(5.516724631660644),
    "adc_b4_ch3": np.float64(4.305328702876725),
    "adc_b4_ch4": np.float64(2.7917000452115617),
    "adc_b4_ch5": np.float64(2.609695937804127),
    "adc_b4_ch6": np.float64(3.295418410414862),
    "adc_b4_ch7": np.float64(4.625765355978418),
    "adc_b4_ch8": np.float64(10.951236980813379),
    "adc_b4_ch9": np.float64(3.164976834647275),
    "adc_b4_ch10": np.float64(4.375403561315023),
    "adc_b4_ch11": np.float64(4.616013355376873),
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


def nn_is_dumb(hitnet_inp, all_chargenet_inp, per_chargenet_inp):
    """Silly NN is a mere computer and thinks all hit times should begin at 60ns. Take the
    min PMT hit time and shift it to 60, shift everything else by that same factor as well.
    However, early dark counts could interfere. So, find median hit time and look behind 20ns
    and forward 20 ns (40ns window, found from plot of summed waveform cumsum)"""
    median_hit_time = np.median(hitnet_inp[3])
    min_hit_time = median_hit_time - 20
    max_hit_time = median_hit_time + 20
    remove_these = [
        i
        for i in range(len(hitnet_inp[3]))
        if (hitnet_inp[3][i] < min_hit_time) or (hitnet_inp[3][i] > max_hit_time)
    ]

    # first make the easy changes to all_chargenet_inp
    for idx in sorted(remove_these, reverse=True):
        del all_chargenet_inp[0][idx]  # delete dark hit charge
    all_chargenet_inp[0] = np.sum(all_chargenet_inp[0])  # reformulate as sum
    all_chargenet_inp[1] -= len(remove_these)  # subtract dark hits

    # now change hitnet_inp
    for idx in sorted(remove_these, reverse=True):
        for sublist in hitnet_inp:
            del sublist[idx]  # delete every entry for dark count
        for sublist in per_chargenet_inp:  # same for this too
            del sublist[idx]

    hit_time_shift = 60 - min(hitnet_inp[3])
    hitnet_inp[3] = [hit_time + hit_time_shift for hit_time in hitnet_inp[3]]
    return hitnet_inp, all_chargenet_inp, per_chargenet_inp


def weighted_average_hit_time(waveform, window_size=10):
    """Do weighted average in window around pulse. Returns float value at which
    hit time occurred in ns (multiply by 2 at end, 500MHz)"""
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
    # accounts for the fact that we are technically returning the sample index of the peak. since
    # sampling rate is 500MHz, we multiply by 2 to get ns
    return (numerator / denominator) * 2


def constant_fraction_time(waveform, fraction=0.5, time_step=1.0):
    """Do leading edge CFD to find when pulse height is 50% of maximum."""
    waveform = np.array(base_and_flip(waveform))
    max_val = np.max(waveform)
    threshold = fraction * max_val

    # Find where signal crosses threshold
    for i in range(1, len(waveform)):
        if waveform[i - 1] < threshold <= waveform[i]:
            # linear interpolation to estimate crossing time
            t0 = (i - 1) * time_step
            y0 = waveform[i - 1]
            y1 = waveform[i]
            crossing_time = t0 + time_step * (threshold - y0) / (y1 - y0)
            return crossing_time
    return None  # no crossing found


def get_channel_charge(waveform):
    """Takes in a raw waveform. Does baseline subtraction, makes it positive, make window of
    size 60ns / 30 sample, integrate by just taking sum (nothing fancy), divide by 50 (resistance),
    returns charge in pC"""
    based_flipped = base_and_flip(waveform)
    time_of_max = np.argmax(based_flipped)
    charge_pC = np.sum(based_flipped[time_of_max - 5 : time_of_max + 5]) / 50
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
    Uses charge to determine if the pulse exceeds threshold or is just noise / fluctuations
    """
    wave_cut = waveform[range_min:range_max]
    if get_channel_charge(wave_cut) > 15:
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
    return b4_ch13_or_ch14_detection_list  # could hit both top paddles


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
        # using argmin here should be fine, we don't need anything too accurate
        peak_sample_time_ns = np.argmin(summed_waveform) * 2  # converts to ns
        # rough estimate of time range
        if 750 < peak_sample_time_ns and i in top_paddle_PMT_events:
            twice_checked_top_paddle_event_index_list.append(i)
    return twice_checked_top_paddle_event_index_list


def get_all_sensor_input(fname: str, peak_method: str):
    """Takes in a file path and a peak_method, which is either CFD or weighted avg.
    Then the information for all-sensor chargenet AND all-sensor hitnet is returned.
    "all" refers to all-sensor and "per" refers to per-sensor. I tried to remember to
    use "total" when talking about everything but sometimes abuse notation and use all
    """

    all_events = []
    per_events = []
    bottom_paddle_tags_list = []

    traces = get_1t_info(fname)[0]
    top_paddle_events = top_paddle_event_list(traces)
    bottom_paddle_1, bottom_paddle_2 = b1_ch0_or_b2_ch15_detections(traces)
    print("BP detections:", bottom_paddle_1, bottom_paddle_2)

    for i in top_paddle_events:
        # print("starting new TP event")
        hitnet_input = [[], [], [], [], []]
        all_chargenet_input = []
        per_chargenet_input = [[], [], [], []]
        temp_chargenet = []
        # sum_of_charges_of_all_hits = 0
        num_of_hits = 0

        # Could have just gotten this intersection with TP events I guess
        bottom_paddle_event_tag = 0
        if (i in bottom_paddle_1) and (i in bottom_paddle_2):
            bottom_paddle_event_tag = 3
        elif i in bottom_paddle_1:
            bottom_paddle_event_tag = 1
        elif i in bottom_paddle_2:
            bottom_paddle_event_tag = 2
        if bottom_paddle_event_tag == 0:  # skip, can't do true comparison
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
            )

            # skip if not a pulse
            # restricting the window could help, but really that's the job of the twice
            # checked TP events function. make sure superposition is proper
            # if not is_pulse(daisy_corrected_waveform): #range_min=750):
            #     continue

            # gets waveform charge and checks if it's a pulse. also add to per_chargenet
            # here since we need everything regardless of if pulse or not

            waveform_charge = get_channel_charge(daisy_corrected_waveform)
            per_chargenet_input[0].append(PMT_location_dict[key][0])
            per_chargenet_input[1].append(PMT_location_dict[key][1])
            per_chargenet_input[2].append(PMT_location_dict[key][2])
            if waveform_charge < 15:  # this is equivalent to is_pulse
                # assume this is dark count and zero, so any charge not due to muon
                per_chargenet_input[3].append(0)
                continue
            per_chargenet_input[3].append(waveform_charge)

            print("hit for", key)
            # hitnet input as we go
            hitnet_input[0].append(PMT_location_dict[key][0])
            hitnet_input[1].append(PMT_location_dict[key][1])
            hitnet_input[2].append(PMT_location_dict[key][2])
            if peak_method == "CFD":
                peak_method_hit_time = constant_fraction_time(
                    daisy_corrected_waveform, fraction=0.5, time_step=2
                )
            elif peak_method == "W_avg":
                peak_method_hit_time = weighted_average_hit_time(
                    daisy_corrected_waveform
                )
            if peak_method_hit_time is None:
                continue  # if peak method fails, continue
            pmt_hit_time = peak_method_hit_time - PMT_channel_delay_dict[key]
            hitnet_input[3].append(pmt_hit_time)
            hitnet_input[4].append(1)

            # chargenet values to later input
            temp_chargenet.append(waveform_charge)
            # sum_of_charges_of_all_hits += waveform_charge
            num_of_hits += 1

        # chargenet input
        all_chargenet_input.append(temp_chargenet)
        all_chargenet_input.append(num_of_hits)

        # reassign to please the picky NN
        hitnet_input, all_chargenet_input, per_chargenet_input = nn_is_dumb(
            hitnet_input, all_chargenet_input, per_chargenet_input
        )
        all_event = {
            "hits": np.stack(hitnet_input, axis=1),
            "total_charge": np.stack(all_chargenet_input),
        }
        per_event = {
            "hits": np.stack(hitnet_input, axis=1),
            "total_charge": np.stack(per_chargenet_input, axis=1),
        }

        # only append if there's anything to reco
        if num_of_hits > 0:
            all_events.append(all_event)
            per_events.append(per_event)
            bottom_paddle_tags_list.append(bottom_paddle_event_tag)
            # print("hitnet and chargent inputs:\n")
            # print(hitnet_input)
            # print(all_chargenet_input)
            # print(per_chargenet_input)
            # print("bp tag", bottom_paddle_event_tag)
            print("\n=== HitNet Input ===")
            for i, sublist in enumerate(hitnet_input):
                print(f"  Sublist {i}: {sublist}")

            print("\n=== All ChargeNet Input ===")
            for i, sublist in enumerate(all_chargenet_input):
                print(f"  Sublist {i}: {sublist}")

            print("\n=== Per-Hit ChargeNet Input ===")
            for i, sublist in enumerate(per_chargenet_input):
                print(f"  Sublist {i}: {sublist}")

            print(f"\n=== Bottom Paddle Event Tag ===\n  {bottom_paddle_event_tag}")

    return all_events, per_events, bottom_paddle_tags_list


# phase_directory = "/media/disk_d/WbLS-DATA/raw_root/phase3/muon/" # Oct 31, 2024
phase_directory = "/media/disk_e/WbLS-DATA/raw_root/phase3/muon/"  # Nov 13, 2024
# phase_directory = "/media/disk_a/WbLS-DATA/raw_root/phase6/muon/" # Jan 07, 2025
# phase_directory = "/media/disk_b/WbLS-DATA/raw_root/phase6/muon/" # Dec 19, 2024
# phase_directory = "/media/disk_e/WbLS-DATA/raw_root/phase4/muon/" # Dec 03, 2024
# phase_directory = "/media/disk_k/WbLS-DATA/raw_root/phase8/muon/" # Mar 11, 2025
file_paths_for_ch_delays = [
    phase_directory + str(f)
    for f in os.listdir(phase_directory)
    if os.path.isfile(os.path.join(phase_directory, f))
]

total_allsensor_events_for_phase = []
total_persensor_events_for_phase = []
total_bottom_paddle_tags_for_phase = []
for fileee in file_paths_for_ch_delays:
    if "water" not in fileee:
        continue  # network was trained on water
    print("starting new file", fileee)
    try:
        allsensor_events, persensor_events, bottom_paddle_tags = get_all_sensor_input(
            fileee, "W_avg"
        )  # change peak method HERE
        total_allsensor_events_for_phase.extend(allsensor_events)
        total_persensor_events_for_phase.extend(persensor_events)
        total_bottom_paddle_tags_for_phase.extend(bottom_paddle_tags)
    except Exception as e:
        print(f"Skipped {fileee} due to error: {e}")

data_to_save = {
    "all_events_for_phase": total_allsensor_events_for_phase,
    "per_events_for_phase": total_persensor_events_for_phase,
    "all_bottom_paddle_tags_for_phase": total_bottom_paddle_tags_for_phase,
}

print(
    "completed:",
    len(total_allsensor_events_for_phase),
    len(total_persensor_events_for_phase),
    len(total_bottom_paddle_tags_for_phase),
)

### THINGS THE USER CAN CHANGE:
# peak method, CFD or Weight
# output pickle file name
# Data for all-sensor and per-sensor is in same file. Make sure you are using numpy 1.24.4 to match the cluster
# scp /media/disk_o/my_pickles/07_07_25_processed_data_disk_e_phase_3_allsensor_CFD_1.pkl dzc5938@submit.hpc.psu.edu:/storage/group/dfc13/default/dcolson/my_pickles

outfile_name = "/media/disk_o/my_pickles/07_09_25_processed_data_disk_e_phase_3_Weight_1.pkl"
with open(
    outfile_name,
    "wb",
) as f:
    pickle.dump(data_to_save, f)
