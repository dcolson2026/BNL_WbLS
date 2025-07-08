import uproot

# import this, lol this is funny
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# these are from paddles and alpha PMT
RELEVANT_CHANNELS = [
    "adc_b1_ch1",
    "adc_b1_ch2",
    "adc_b1_ch3",
    "adc_b1_ch4",
    "adc_b1_ch5",
    "adc_b1_ch6",
    "adc_b1_ch7",
    "adc_b1_ch8",
    "adc_b1_ch9",
    "adc_b1_ch10",
    "adc_b1_ch11",
    "adc_b1_ch12",
    "adc_b1_ch13",
    "adc_b1_ch14",
    "adc_b1_ch15",
    "adc_b2_ch0",
    "adc_b2_ch1",
    "adc_b2_ch2",
    "adc_b2_ch3",
    "adc_b2_ch4",
    "adc_b2_ch5",
    "adc_b2_ch6",
    "adc_b2_ch7",
    "adc_b2_ch8",
    "adc_b2_ch9",
    "adc_b2_ch10",
    "adc_b2_ch11",
    "adc_b2_ch12",
    "adc_b2_ch13",
    "adc_b2_ch14",
    "adc_b3_ch0",
    "adc_b3_ch1",
    "adc_b3_ch2",
    "adc_b3_ch3",
    "adc_b3_ch4",
    "adc_b3_ch5",
    "adc_b3_ch6",
    "adc_b3_ch7",
    "adc_b3_ch8",
    "adc_b3_ch9",
    "adc_b3_ch10",
    "adc_b3_ch11",
    "adc_b3_ch12",
    "adc_b3_ch13",
    "adc_b3_ch14",
    "adc_b3_ch15",
    "adc_b4_ch0",
    "adc_b4_ch1",
    "adc_b4_ch2",
    "adc_b4_ch3",
    "adc_b4_ch4",
    "adc_b4_ch5",
    "adc_b4_ch6",
    "adc_b4_ch7",
    "adc_b4_ch8",
    "adc_b4_ch9",
    "adc_b4_ch10",
    "adc_b4_ch11",
]
IRRELEVANT_CHANNELS = [
    "adc_b1_ch0",
    "adc_b2_ch15",
    "adc_b4_ch12",
    "adc_b4_ch13",
    "adc_b4_ch14",
    "adc_b4_ch15",
]

# 58 TOTAL PMTs
BOTTOM_PMT_CHANNELS = [
    "adc_b1_ch1",
    "adc_b1_ch2",
    "adc_b1_ch3",
    "adc_b1_ch4",
    "adc_b1_ch5",
    "adc_b1_ch6",
    "adc_b1_ch7",
    "adc_b1_ch8",
    "adc_b1_ch9",
    "adc_b1_ch10",
    "adc_b1_ch11",
    "adc_b1_ch12",
    "adc_b1_ch13",
    "adc_b1_ch14",
    "adc_b1_ch15",
    "adc_b2_ch0",
    "adc_b2_ch1",
    "adc_b2_ch2",
    "adc_b2_ch3",
    "adc_b2_ch4",
    "adc_b2_ch5",
    "adc_b2_ch6",
    "adc_b2_ch7",
    "adc_b2_ch8",
    "adc_b2_ch9",
    "adc_b2_ch10",
    "adc_b2_ch11",
    "adc_b2_ch12",
    "adc_b2_ch13",
    "adc_b2_ch14",
]
# all side, including supp (28)
SIDE_PMT_CHANNELS = [
    "adc_b3_ch0",
    "adc_b3_ch1",
    "adc_b3_ch2",
    "adc_b3_ch3",
    "adc_b3_ch4",
    "adc_b3_ch5",
    "adc_b3_ch6",
    "adc_b3_ch7",
    "adc_b3_ch8",
    "adc_b3_ch9",
    "adc_b3_ch10",
    "adc_b3_ch11",
    "adc_b3_ch12",
    "adc_b3_ch13",
    "adc_b3_ch14",
    "adc_b3_ch15",
    "adc_b4_ch0",
    "adc_b4_ch1",
    "adc_b4_ch2",
    "adc_b4_ch3",
    "adc_b4_ch4",
    "adc_b4_ch5",
    "adc_b4_ch6",
    "adc_b4_ch7",
    "adc_b4_ch8",
    "adc_b4_ch9",
    "adc_b4_ch10",
    "adc_b4_ch11",
]

# produced using /home/dcolson/my_analysis/tof_alpha_source.py
CHANNELS_AND_DISTANCES_MM_DICT = {
    "adc_b1_ch1": np.float64(849.099648156799),
    "adc_b1_ch2": np.float64(831.7868912768462),
    "adc_b1_ch3": np.float64(829.9985737939554),
    "adc_b1_ch4": np.float64(843.8334032852694),
    "adc_b1_ch5": np.float64(817.181418046201),
    "adc_b1_ch6": np.float64(774.2681835126638),
    "adc_b1_ch7": np.float64(746.5426645544112),
    "adc_b1_ch8": np.float64(735.7239020175979),
    "adc_b1_ch9": np.float64(742.5512440229294),
    "adc_b1_ch10": np.float64(766.5533380006899),
    "adc_b1_ch11": np.float64(806.1976618671131),
    "adc_b1_ch12": np.float64(799.0517583360918),
    "adc_b1_ch13": np.float64(746.4080469153586),
    "adc_b1_ch14": np.float64(708.4456312943147),
    "adc_b1_ch15": np.float64(687.6004890196051),
    "adc_b2_ch0": np.float64(685.4360892891474),
    "adc_b2_ch1": np.float64(702.1252114117539),
    "adc_b2_ch2": np.float64(736.3871077768812),
    "adc_b2_ch3": np.float64(785.9269129505618),
    "adc_b2_ch4": np.float64(770.6305664843563),
    "adc_b2_ch5": np.float64(724.9670475269893),
    "adc_b2_ch6": np.float64(695.2783255646619),
    "adc_b2_ch7": np.float64(683.6487840989699),
    "adc_b2_ch8": np.float64(690.9908465385051),
    "adc_b2_ch9": np.float64(716.7217172654949),
    "adc_b2_ch10": np.float64(758.973431682559),
    "adc_b2_ch11": np.float64(757.0615645375216),
    "adc_b2_ch12": np.float64(737.5916434586281),
    "adc_b2_ch13": np.float64(735.5743555209086),
    "adc_b2_ch14": np.float64(751.1503261664739),
    "adc_b3_ch0": np.float64(659.9604927758934),
    "adc_b3_ch1": np.float64(550.9007755712457),
    "adc_b3_ch2": np.float64(489.51080123425265),
    "adc_b3_ch3": np.float64(587.5588923886694),
    "adc_b3_ch4": np.float64(801.4576670199119),
    "adc_b3_ch5": np.float64(714.3375984259824),
    "adc_b3_ch6": np.float64(668.1372347991093),
    "adc_b3_ch7": np.float64(742.9750951579737),
    "adc_b3_ch8": np.float64(743.5038345731648),
    "adc_b3_ch9": np.float64(648.646101140676),
    "adc_b3_ch10": np.float64(597.3867461912761),
    "adc_b3_ch11": np.float64(680.05555069053),
    "adc_b3_ch12": np.float64(724.6270019982695),
    "adc_b3_ch13": np.float64(626.9195359254647),
    "adc_b3_ch14": np.float64(573.7222886771962),
    "adc_b3_ch15": np.float64(659.3647640153363),
    "adc_b4_ch0": np.float64(524.24994644984),
    "adc_b4_ch1": np.float64(481.6652557938034),
    "adc_b4_ch2": np.float64(496.8615186726177),
    "adc_b4_ch3": np.float64(633.6100188228246),
    "adc_b4_ch4": np.float64(598.8531274351918),
    "adc_b4_ch5": np.float64(611.1423879405355),
    "adc_b4_ch6": np.float64(648.8899545783246),
    "adc_b4_ch7": np.float64(614.9972239278077),
    "adc_b4_ch8": np.float64(626.9701233214068),
    "adc_b4_ch9": np.float64(505.2143002258151),
    "adc_b4_ch10": np.float64(460.8740624497218),
    "adc_b4_ch11": np.float64(476.7335225654475),
}

# updated to add refractive index of 1.33
CHANNELS_AND_TIMES_NS_DICT = {
    "adc_b1_ch1": np.float64(3.7643417734951425),
    "adc_b1_ch2": np.float64(3.6875885513273516),
    "adc_b1_ch3": np.float64(3.679660343819869),
    "adc_b1_ch4": np.float64(3.740994754564694),
    "adc_b1_ch5": np.float64(3.6228376200048245),
    "adc_b1_ch6": np.float64(3.432588946906143),
    "adc_b1_ch7": np.float64(3.3096724795245565),
    "adc_b1_ch8": np.float64(3.2617092989446843),
    "adc_b1_ch9": np.float64(3.291977181834987),
    "adc_b1_ch10": np.float64(3.3983864651363924),
    "adc_b1_ch11": np.float64(3.574142967610868),
    "adc_b1_ch12": np.float64(3.542462795290007),
    "adc_b1_ch13": np.float64(3.30907567465809),
    "adc_b1_ch14": np.float64(3.140775632071462),
    "adc_b1_ch15": np.float64(3.0483621679869164),
    "adc_b2_ch0": np.float64(3.0387666625152203),
    "adc_b2_ch1": np.float64(3.112755103925442),
    "adc_b2_ch2": np.float64(3.2646495111441736),
    "adc_b2_ch3": np.float64(3.4842759807474906),
    "adc_b2_ch4": np.float64(3.4164621780806463),
    "adc_b2_ch5": np.float64(3.2140205773696526),
    "adc_b2_ch6": np.float64(3.0824005766700013),
    "adc_b2_ch7": np.float64(3.030842942838767),
    "adc_b2_ch8": np.float64(3.0633927529873732),
    "adc_b2_ch9": np.float64(3.1774662798770277),
    "adc_b2_ch10": np.float64(3.364782213792678),
    "adc_b2_ch11": np.float64(3.356306269449679),
    "adc_b2_ch12": np.float64(3.269989619333251),
    "adc_b2_ch13": np.float64(3.2610463094760287),
    "adc_b2_ch14": np.float64(3.3300997793380347),
    "adc_b3_ch0": np.float64(2.925824851306461),
    "adc_b3_ch1": np.float64(2.4423267716991894),
    "adc_b3_ch2": np.float64(2.1701645521385204),
    "adc_b3_ch3": np.float64(2.604844422923101),
    "adc_b3_ch4": np.float64(3.553128990454943),
    "adc_b3_ch5": np.float64(3.166896686355189),
    "adc_b3_ch6": np.float64(2.9620750742760515),
    "adc_b3_ch7": np.float64(3.29385625520035),
    "adc_b3_ch8": np.float64(3.296200333274364),
    "adc_b3_ch9": np.float64(2.875664381723664),
    "adc_b3_ch10": np.float64(2.648414574781324),
    "adc_b3_ch11": np.float64(3.0149129413946834),
    "adc_b3_ch12": np.float64(3.2125130421923282),
    "adc_b3_ch13": np.float64(2.779343275936227),
    "adc_b3_ch14": np.float64(2.5435021464689034),
    "adc_b3_ch15": np.float64(2.9231837871346573),
    "adc_b4_ch0": np.float64(2.3241747625942906),
    "adc_b4_ch1": np.float64(2.1353826340191953),
    "adc_b4_ch2": np.float64(2.2027527327819385),
    "adc_b4_ch3": np.float64(2.8090044167811894),
    "adc_b4_ch4": np.float64(2.6549155316293507),
    "adc_b4_ch5": np.float64(2.7093979198697076),
    "adc_b4_ch6": np.float64(2.8767454652972395),
    "adc_b4_ch7": np.float64(2.7264876927466144),
    "adc_b4_ch8": np.float64(2.7795675467249037),
    "adc_b4_ch9": np.float64(2.2397833976677806),
    "adc_b4_ch10": np.float64(2.0432083435271),
    "adc_b4_ch11": np.float64(2.1135186167068176),
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


def correct_times(event_ttt1, event_ttt5, event_id):
    """Correct the events by comparing closest in 1 and 5 board"""
    array_idx = sorted(range(len(event_id)), key=lambda i: event_id[i])

    event_ttt5_good = event_ttt5[array_idx]
    event_ttt1_good = event_ttt1[array_idx]

    event_ttt5_good_idx = []
    event_ttt1_good_idx = []

    window_size = 3

    for i, val1 in enumerate(event_ttt1_good):
        # Define the search window (max 3 elements before and after in event_ttt5_good)
        start_idx = max(i - window_size, 0)
        end_idx = min(i + window_size + 1, len(event_ttt5_good))

        # Find the index of the closest element in event_ttt5_good within the window
        local_window = event_ttt5_good[start_idx:end_idx]
        closest_idx = (
            np.argmin(np.abs(local_window - val1)) + start_idx
        )  # Add start_idx to get global index

        if -17 < (event_ttt5_good[closest_idx] - val1) < -13:
            event_ttt1_good_idx.append(i)
            event_ttt5_good_idx.append(closest_idx)

    event_ttt1_good_final = np.array(array_idx)[event_ttt1_good_idx]
    event_ttt5_good_final = np.array(array_idx)[event_ttt5_good_idx]

    return event_ttt1_good_final, event_ttt5_good_final


def write_corrected_root(
    outfname,
    traces,
    event_ttt1,
    event_ttt2,
    event_ttt3,
    event_ttt4,
    event_ttt5,
    event_id,
    event_sanity,
    daqkeys,
    run_info,
    event_ttt1_good_final,
    event_ttt5_good_final,
):

    new_daq = {}
    for key in daqkeys:
        if "adc_b5" in key:
            new_daq[key] = traces[key][event_ttt5_good_final]
        elif "adc" in key:
            new_daq[key] = traces[key][event_ttt1_good_final]
    new_daq["event_ttt_5"] = event_ttt5[event_ttt5_good_final]
    new_daq["event_ttt_4"] = event_ttt4[event_ttt1_good_final]
    new_daq["event_ttt_3"] = event_ttt3[event_ttt1_good_final]
    new_daq["event_ttt_2"] = event_ttt2[event_ttt1_good_final]
    new_daq["event_ttt_1"] = event_ttt1[event_ttt1_good_final]
    new_daq["event_id"] = event_id[event_ttt1_good_final]
    new_daq["event_sanity"] = event_sanity[event_ttt1_good_final]

    output_file = uproot.recreate(outfname)
    output_file["daq"] = {branch: new_daq[branch] for branch in new_daq}

    if run_info is not None:
        data = run_info.arrays(library="np")
        output_file["run_info"] = {branch: data[branch] for branch in data}
    output_file.close()


def waveform_daisy_correction(waveform, boardID):
    if (boardID < 1) or (boardID > 4):
        print("Bad BoardID")
        return False
    elif boardID != 1:
        return waveform[24 * (4 - boardID) : -24 * (boardID - 1)]
    else:
        return waveform[24 * 3 :]


def need_event_mismatch_correction(fname: str):
    """This function breaks at year 2100. But so does this file convention naming anyways."""
    temp1 = fname.split("/")[-1]  # gets the actual fname from the path
    temp2 = temp1.split("_")
    date_string = temp2[2][:6]
    if 241017 <= int(date_string) <= 250329:
        print("event mismatch will be done for " + fname)
        return True
    return False


def quickly_correct_file(fname: str, outfname: str) -> str:
    """Does only event mismatch correction, and only if needed."""
    if not need_event_mismatch_correction(fname):
        return fname
    (
        file_traces,
        event_ttt1,
        event_ttt2,
        event_ttt3,
        event_ttt4,
        event_ttt5,
        file_event_ids,
        file_event_sanity,
        file_daqkeys,
        file_run_info,
    ) = get_1t_info(fname)
    mismatch_corrected_event_ttt1, mismatch_corrected_event_ttt5 = correct_times(
        event_ttt1, event_ttt5, file_event_ids
    )
    write_corrected_root(
        outfname,
        file_traces,
        event_ttt1,
        event_ttt2,
        event_ttt3,
        event_ttt4,
        event_ttt5,
        file_event_ids,
        file_event_sanity,
        file_daqkeys,
        file_run_info,
        mismatch_corrected_event_ttt1,
        mismatch_corrected_event_ttt5,
    )
    print("ROOT file corrected for " + fname)
    return outfname


def b4_ch12_detections(traces) -> list[int]:
    """Returns a list of the event indices where the alpha PMT goes off. The waveform is not altered at all prior to this.
    In other words, we are purely looking at the shape of the waveform."""
    b4_ch12_detection_list = []
    waveforms_list = traces["adc_b4_ch12"]
    for i, waveform in enumerate(waveforms_list):
        if is_pulse(waveform):  # this is arbitrary, and hopefully this is sufficient
            b4_ch12_detection_list.append(i)
    return b4_ch12_detection_list


def alpha_event_list(traces) -> list[int]:
    """Returns a list of event indices that correspond to alpha particle events.
    This means b4_ch12 has a signal and the superposition of signals for that event
    lies in the time range for alpha events."""

    twice_checked_alpha_event_index_list = []
    num_events = len(traces["adc_b2_ch1"])  # pick arbitrary PMT, all same length
    alpha_PMT_events = b4_ch12_detections(traces)

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
        peak_sample_time_ns = np.argmin(summed_waveform) * 2  # converts ADU to mV
        # rough estimate of time range
        if 550 < peak_sample_time_ns < 750 and i in alpha_PMT_events:
            twice_checked_alpha_event_index_list.append(i)
    return twice_checked_alpha_event_index_list


def peak_of_waveform_sum_in_event(traces):
    """Calculates superposition of waveforms from relevant PMTs per event.
    Takes TIME of peak value of superposition and adds to list."""

    peak_sample_time_list = []
    num_events = len(traces["adc_b2_ch1"])  # pick arbitrary PMT, all same length

    for i in range(num_events):
        corrected_waveforms_per_event = []
        # waveform loop to get the i_th waveform for each PMT
        for key in traces.keys():
            if ("b5" in key) or (key in IRRELEVANT_CHANNELS):
                continue
            board_num = int(key[5])
            uncorrected_waveform = traces[key][i]
            corrected_waveforms_per_event.append(
                waveform_daisy_correction(uncorrected_waveform, board_num)
            )
        summed_waveform = np.sum(corrected_waveforms_per_event, axis=0)
        peak_sample_time_ns = np.argmin(summed_waveform) * 2  # converts ADU to mV
        peak_sample_time_list.append(peak_sample_time_ns)
    return peak_sample_time_list


def get_channel_delays(traces):
    """For every alpha event, a dict is made with keys as PMT channels and values as delays.
    This dict is appended to a list for every event, and the list is returned."""
    channel_delays = []
    twice_checked_alpha_event_index_list = alpha_event_list(traces)
    for event_i in twice_checked_alpha_event_index_list:
        event_channel_delays_ns_dict = {}
        # daisy correction, finds time of pulse, converts alpha hit time in ns
        alpha_hit_time_i = (
            np.argmin(waveform_daisy_correction(traces["adc_b4_ch12"][event_i], 4)) * 2
        )
        event_channel_delays_ns_dict["adc_b4_ch12"] = alpha_hit_time_i

        # waveform loop to get the i_th waveform for each PMT
        for key in traces.keys():
            if ("b5" in key) or (key in IRRELEVANT_CHANNELS):
                continue
            board_num = int(key[5])
            corrected_waveform = waveform_daisy_correction(
                traces[key][event_i], board_num
            )
            # if is_pulse(corrected_waveform, 550 // 2, 750 // 2):
            # is_pulse uses sample index and alpha_hit_time_i is in ns, oops
            if is_pulse(
                corrected_waveform,
                (alpha_hit_time_i - 20) // 2,
                (alpha_hit_time_i + 40) // 2,
            ):
                pmt_hit_time = weighted_average_hit_time(corrected_waveform) * 2
                event_channel_delays_ns_dict[key] = (
                    pmt_hit_time - alpha_hit_time_i - CHANNELS_AND_TIMES_NS_DICT[key]
                )
            else:
                event_channel_delays_ns_dict[key] = None

        channel_delays.append(event_channel_delays_ns_dict)
    return channel_delays


def base_and_flip(waveform):
    """Subtract baseline and reflect over x axis"""
    positive_waveform = (waveform - np.median(waveform)) * (-1)
    return positive_waveform


def get_channel_charge(waveform):
    """Takes in a raw waveform. Does baseline subtraction, makes it positive, make window of
    size 60ns / 30 sample, integrate by just taking sum (nothing fancy), divide by 50 (resistance),
    returns charge in pC"""
    based_flipped = base_and_flip(waveform)
    time_of_max = np.argmax(based_flipped)
    charge_pC = np.sum(based_flipped[time_of_max - 5 : time_of_max + 5]) / 50
    return charge_pC


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


# def is_pulse(waveform: np.ndarray, range_min: int = 0, range_max: int = 1928) -> bool:
#     """Takes in a daisy corrected waveform and looks in a given range to see if there is a pulse.
#     For example, you can use some range around an alpha PMT hit if looking for just alpha detections.
#     """
#     threshold_sigma = 20
#     # this is arbitrary, and hopefully sufficient
#     baseline = np.median(waveform[: int(0.5 * len(waveform))])
#     noise_std = np.std(waveform[: int(0.5 * len(waveform))])
#     deviation = np.abs(waveform - baseline)
#     threshold = threshold_sigma * noise_std

#     # Get all indices where deviation exceeds threshold
#     pulse_indices = np.where(deviation > threshold)[0]
#     for i in pulse_indices:
#         if range_min <= i <= range_max:
#             return True
#     return False


def is_pulse(waveform: np.ndarray, range_min: int = 0, range_max: int = 1928) -> bool:
    """Takes in a daisy corrected waveform and looks in a given range to see if there is a pulse.
    For example, you can use some range around an alpha PMT hit if looking for just alpha detections.
    Uses charge to determine if the pulse exceeds threshold or is just noise / fluctuations
    """
    wave_cut = waveform[range_min:range_max]
    if get_channel_charge(wave_cut) > 15:
        return True
    return False


def b4_ch13_or_ch14_detections(traces):
    """These are top paddle channels, gets list of events with detections."""
    b4_ch13_or_ch14_detection_list = []
    waveforms_list = traces["adc_b4_ch13"] + traces["adc_b4_ch14"]
    for i, waveform in enumerate(waveforms_list):
        if is_pulse(waveform):  # this is arbitrary, and hopefully this is sufficient
            b4_ch13_or_ch14_detection_list.append(i)
    return b4_ch13_or_ch14_detection_list


def b1_ch0_or_b2_ch15(traces):
    """These are bottom paddle channels, gets list of events with detections."""
    b4_ch13_or_ch14_detection_list = []
    waveforms_list = traces["adc_b1_ch0"] + traces["adc_b2_ch15"]
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
        peak_sample_time_ns = np.argmin(summed_waveform) * 2  # converts ADU to mV
        # rough estimate of time range
        if 750 < peak_sample_time_ns and i in top_paddle_PMT_events:
            twice_checked_top_paddle_event_index_list.append(i)
    return twice_checked_top_paddle_event_index_list


def more_than_26_bottom(traces) -> list[int]:
    """Returns of a list of event indices where 26 or more bottom PMTs
    detect a pulse"""
    pass


def extract_gains(csv_file_path: str) -> dict[float]:
    """Opens the csv file and extracts the channel as keys and spe mean as value.
    Improve this in future by having fname as the argument and have this function
    find the nearest csv from that"""
    gains_dict = {}
    with open(csv_file_path, mode="r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines[1:]:
        # Split CSV line into fields
        fields = line.strip().split(",")
        pmt_channel = str(fields[1])
        gains_dict[pmt_channel] = float(fields[3])
    return gains_dict


def median_and_error(array):
    """Returns median and its standard error"""
    cleaned_array = [i for i in array if i is not None]
    median = np.median(cleaned_array)

    # Compute the standard deviation (unbiased, ddof=1)
    std_dev = np.std(cleaned_array, ddof=1)

    # Number of samples
    n = len(cleaned_array)

    # Standard error of the mean
    sem = std_dev / np.sqrt(n)

    # Approximate standard error of the median
    sem_median = 1.25 * sem

    return median, sem_median


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    phase_directory = "/media/disk_d/WbLS-DATA/raw_root/phase3/muon/"  # Oct 31, 2024
    # phase_directory = "/media/disk_e/WbLS-DATA/raw_root/phase3/muon/"  # Nov 13, 2024
    # phase_directory = "/media/disk_a/WbLS-DATA/raw_root/phase6/muon/" # Jan 07, 2025
    # phase_directory = "/media/disk_b/WbLS-DATA/raw_root/phase6/muon/" # Dec 19, 2024
    # phase_directory = "/media/disk_e/WbLS-DATA/raw_root/phase4/muon/"  # Dec 03, 2024
    # phase_directory = "/media/disk_k/WbLS-DATA/raw_root/phase8/muon/" # Mar 11, 2025
    file_paths_for_ch_delays = [
        phase_directory + str(f)
        for f in os.listdir(phase_directory)
        if os.path.isfile(os.path.join(phase_directory, f))
    ]
    ch_delays_for_a_ch_dict = {key: [] for key in RELEVANT_CHANNELS}

    # get channel delays for multiple files
    for fpath in file_paths_for_ch_delays:
        print("now doing", fpath)
        # corrected_file_name = "/media/disk_o/my_corrected_roots/disk_d_phase_3/corrected_" + fpath.split("/")[-1]
        # quickly_correct_file(
        #     fpath,
        #     corrected_file_name,
        # )
        try:
            (
                file_traces,
                event_ttt1,
                event_ttt2,
                event_ttt3,
                event_ttt4,
                event_ttt5,
                file_event_ids,
                file_event_sanity,
                file_daqkeys,
                file_run_info,
            ) = get_1t_info(fpath)

            ch_delay_list_per_file = get_channel_delays(file_traces)
            for alpha_event_dict in ch_delay_list_per_file:
                for key in alpha_event_dict:
                    if key in RELEVANT_CHANNELS:
                        ch_delays_for_a_ch_dict[key].append(alpha_event_dict[key])
        except Exception as e:
            print(f"Skipped {fpath} due to error: {e}")

    median_dict = {key: None for key in RELEVANT_CHANNELS}
    median_error_dict = {key: None for key in RELEVANT_CHANNELS}
    for key, value in ch_delays_for_a_ch_dict.items():
        median, ste_median = median_and_error(value)
        median_dict[key] = median
        median_error_dict[key] = ste_median

    print(median_dict)
