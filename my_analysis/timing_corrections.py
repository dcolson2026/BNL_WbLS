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
# all side, including supp (28 total)
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

# produced using /home/dcolson/my_analysis/TOF_alpha_source.py
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
CHANNELS_AND_TIMES_NS_DICT = {
    "adc_b1_ch1": np.float64(2.8303321605226635),
    "adc_b1_ch2": np.float64(2.7726229709228205),
    "adc_b1_ch3": np.float64(2.766661912646518),
    "adc_b1_ch4": np.float64(2.8127780109508977),
    "adc_b1_ch5": np.float64(2.723938060154003),
    "adc_b1_ch6": np.float64(2.5808939450422126),
    "adc_b1_ch7": np.float64(2.488475548514704),
    "adc_b1_ch8": np.float64(2.4524130067253265),
    "adc_b1_ch9": np.float64(2.4751708134097647),
    "adc_b1_ch10": np.float64(2.555177793335633),
    "adc_b1_ch11": np.float64(2.6873255395570435),
    "adc_b1_ch12": np.float64(2.663505861120306),
    "adc_b1_ch13": np.float64(2.4880268230511953),
    "adc_b1_ch14": np.float64(2.3614854376477155),
    "adc_b1_ch15": np.float64(2.2920016300653505),
    "adc_b2_ch0": np.float64(2.284786964297158),
    "adc_b2_ch1": np.float64(2.3404173713725127),
    "adc_b2_ch2": np.float64(2.454623692589604),
    "adc_b2_ch3": np.float64(2.6197563765018725),
    "adc_b2_ch4": np.float64(2.5687685549478543),
    "adc_b2_ch5": np.float64(2.4165568250899643),
    "adc_b2_ch6": np.float64(2.317594418548873),
    "adc_b2_ch7": np.float64(2.2788292803298997),
    "adc_b2_ch8": np.float64(2.3033028217950173),
    "adc_b2_ch9": np.float64(2.389072390884983),
    "adc_b2_ch10": np.float64(2.529911438941863),
    "adc_b2_ch11": np.float64(2.523538548458405),
    "adc_b2_ch12": np.float64(2.4586388115287603),
    "adc_b2_ch13": np.float64(2.451914518403029),
    "adc_b2_ch14": np.float64(2.503834420554913),
    "adc_b3_ch0": np.float64(2.199868309252978),
    "adc_b3_ch1": np.float64(1.836335918570819),
    "adc_b3_ch2": np.float64(1.6317026707808422),
    "adc_b3_ch3": np.float64(1.9585296412955646),
    "adc_b3_ch4": np.float64(2.6715255567330396),
    "adc_b3_ch5": np.float64(2.381125328086608),
    "adc_b3_ch6": np.float64(2.227124115997031),
    "adc_b3_ch7": np.float64(2.476583650526579),
    "adc_b3_ch8": np.float64(2.4783461152438826),
    "adc_b3_ch9": np.float64(2.16215367046892),
    "adc_b3_ch10": np.float64(1.9912891539709203),
    "adc_b3_ch11": np.float64(2.2668518356351),
    "adc_b3_ch12": np.float64(2.4154233399942315),
    "adc_b3_ch13": np.float64(2.089731786418216),
    "adc_b3_ch14": np.float64(1.9124076289239873),
    "adc_b3_ch15": np.float64(2.1978825467177874),
    "adc_b4_ch0": np.float64(1.7474998214994666),
    "adc_b4_ch1": np.float64(1.6055508526460114),
    "adc_b4_ch2": np.float64(1.656205062242059),
    "adc_b4_ch3": np.float64(2.112033396076082),
    "adc_b4_ch4": np.float64(1.9961770914506394),
    "adc_b4_ch5": np.float64(2.0371412931351185),
    "adc_b4_ch6": np.float64(2.162966515261082),
    "adc_b4_ch7": np.float64(2.0499907464260256),
    "adc_b4_ch8": np.float64(2.089900411071356),
    "adc_b4_ch9": np.float64(1.6840476674193838),
    "adc_b4_ch10": np.float64(1.536246874832406),
    "adc_b4_ch11": np.float64(1.589111741884825),
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
            np.argmin(waveform_daisy_correction(traces["adc_b4_ch12"][event_i], 4))
            * 2
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
            if is_pulse(corrected_waveform, (alpha_hit_time_i - 20) // 2, (alpha_hit_time_i + 40) // 2):
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
    charge_pC = np.sum(based_flipped[time_of_max - 15: time_of_max + 15]) / 50
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

def is_pulse(waveform: np.ndarray, range_min: int = 0, range_max: int = 1928) -> bool:
    """Takes in a daisy corrected waveform and looks in a given range to see if there is a pulse.
    For example, you can use some range around an alpha PMT hit if looking for just alpha detections.
    """
    threshold_sigma = 20
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
    # phase_directory = "/media/disk_d/WbLS-DATA/raw_root/phase3/muon/" # Oct 31, 2024
    # phase_directory = "/media/disk_e/WbLS-DATA/raw_root/phase3/muon/"  # Nov 13, 2024
    # phase_directory = "/media/disk_a/WbLS-DATA/raw_root/phase6/muon/" # Jan 07, 2025
    # phase_directory = "/media/disk_b/WbLS-DATA/raw_root/phase6/muon/" # Dec 19, 2024
    phase_directory = "/media/disk_e/WbLS-DATA/raw_root/phase4/muon/" # Dec 03, 2024
    # phase_directory = "/media/disk_k/WbLS-DATA/raw_root/phase8/muon/" # Mar 11, 2025
    file_paths_for_ch_delays = [phase_directory + str(f) for f in os.listdir(phase_directory) if os.path.isfile(os.path.join(phase_directory, f))]
    ch_delays_for_a_ch_dict = {key: [] for key in RELEVANT_CHANNELS}

    # file_paths_for_ch_delays = [
    #     "/media/disk_a/WbLS-DATA/raw_root/phase6/muon/muon_wbls05_250105T0911_127.root",
    #     "/media/disk_a/WbLS-DATA/raw_root/phase7/muon/muon_wbls06_250111T1142_7.root",
    #     "/media/disk_b/WbLS-DATA/raw_root/phase5/muon/muon_wbls04_241216T1018_68.root",
    #     "/media/disk_d/WbLS-DATA/raw_root/phase3/muon/muon_water_241106T0956_84.root",
    #     "/media/disk_l/WbLS-DATA/raw_root/phase9/muon/muon_wbls1pct_250327T1002_97.root",
    # ]
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


    # # Create the histograms
    # for key, value in ch_delays_for_a_ch_dict.items():
    #     plt.figure(figsize=(8, 6))
    #     plt.hist([i for i in value if i is not None], bins=10, edgecolor="black")
    #     plt.title("Channel Delays for " + key)
    #     plt.xlabel("Delay in ns")
    #     plt.ylabel("Counts")
    #     # plt.yscale("log")
    #     # plt.xlim(0, 100)
    #     # plt.xticks(np.arange(0, 1001, 50))

    #     # Save the figure
    #     plt.savefig(f"/media/disk_o/my_histograms/disk_d_phase_3/{key}_delays.pdf")
    #     plt.close()


    # ORDERS OF CORRECTION
    # first correct event mismatch
    # then do daisy chain effects
    # then alpha timing (minute adjustment)

    # Okay, we can pretty easily make some kind of loop that goes through every file and corrects it if it matches the dating criteria
    # Now let's look at the daisy chain effect in our /home/dcolson/my_analysis/corrected_file_test.root

    # MAKE CORRECT FILE
    # quickly_correct_file("/media/disk_a/WbLS-DATA/raw_root/phase6/muon/muon_wbls05_250102T0921_65.root", f"test{test_num}.root")
    # quickly_correct_file("/media/disk_a/WbLS-DATA/raw_root/phase7/muon/muon_wbls06_250112T1041_97.root", f"/media/disk_o/my_corrected_roots/test{test_num}.root")
    # quickly_correct_file(
    #     "/media/disk_c/WbLS-DATA/raw_root/phase3/muon/muon_water_241020T0943_91.root",
    #     f"/media/disk_o/my_corrected_roots/test{test_num}.root",
    # )

    # ANALYZING CORRECTED FILE
    # corrected_file_name = f"/media/disk_o/my_corrected_roots/test{test_num}.root"
    # (
    #     file_traces,
    #     event_ttt1,
    #     event_ttt2,
    #     event_ttt3,
    #     event_ttt4,
    #     event_ttt5,
    #     file_event_ids,
    #     file_event_sanity,
    #     file_daqkeys,
    #     file_run_info,
    # ) = get_1t_info(corrected_file_name)

    # gains_dict = extract_gains("/media/disk_a/WbLS-DATA/csv/phase6/bnl1t_spe_fit_results_250102.csv")

    """AHHHH"""

    # #####
    # # event loop
    # num_events = len(file_traces["adc_b2_ch1"])
    # the_alpha_events = alpha_event_list(file_traces)
    # the_top_paddle_events = top_paddle_event_list(file_traces)
    # # throughgoing_events = list(set(b4_ch13_or_ch14_detections) + set(b1_ch0_or_b2_ch15))

    # # set logic to get majority list
    # set_alpha = set(the_alpha_events)
    # set_top_paddle = set(the_top_paddle_events)
    # set_alpha_and_TP = set_alpha.union(set_top_paddle)
    # set_temp = set([x for x in range(num_events)])
    # the_majority_events = list(set_temp - set_alpha_and_TP)

    # ratio_side_to_bottom_PE_list = []
    # total_PE_per_event_list = []

    # # event loop, iterate through all events
    # for i in range(num_events):
    #     bottom_PE = 0
    #     side_PE = 0

    #     # PMT channel loop to iterate through channels
    #     for key in file_traces.keys():
    #         if ("b5" in key) or (key in IRRELEVANT_CHANNELS):
    #             continue

    #         board_num = int(key[5])
    #         # correct the daisy chain as a function of board number
    #         corrected_waveform = waveform_daisy_correction(file_traces[key][i], board_num)
    #         # do rudimentary baseline subtraction and reflect across x axis
    #         corrected_waveform = (corrected_waveform - np.median(corrected_waveform))* (-1)
    #         # also a rudimentary but effective peak indx finder
    #         peak_indx = np.argmax(corrected_waveform)
    #         # make windows so that we integrate around the peak just to be careful I thought it might help
    #         lower_window = peak_indx - 20
    #         upper_window = peak_indx + 20

    #         if key in BOTTOM_PMT_CHANNELS:
    #             bottom_PE += np.sum(corrected_waveform[lower_window:upper_window]) / gains_dict[key]
    #         elif key in SIDE_PMT_CHANNELS:
    #             side_PE += np.sum(corrected_waveform[lower_window:upper_window]) / gains_dict[key]
    #     if bottom_PE != 0:
    #         ratio_side_to_bottom_PE_list.append(side_PE / bottom_PE)
    #     else:
    #         print("bad div by zero")
    #     total_PE_per_event_list.append(side_PE + bottom_PE)
    # #####

    ####
    # MAKE MATPLOT

    # multiple plots

    # # Create the histogram
    # plt.figure(figsize=(8, 6))
    # plt.hist(total_PE_per_event_list, bins="auto", edgecolor="black")
    # plt.title("Total PE per Event")
    # plt.xlabel("PE")
    # plt.ylabel("Counts")
    # # plt.yscale("log")
    # plt.xlim(0, 0.5e6)
    # # plt.legend()
    # # plt.xticks(np.arange(0, 1001, 50))

    # # Save the figure
    # plt.savefig("/media/disk_o/my_histograms/total_PE_per_event.pdf")
    # plt.close()

    ####

    # alpha_PMT_events = b4_ch12_detections(file_traces)

    # # Sunwoo says board 1 is "master board", so there shouldn't be a 48ns delay for that one
    # # Now we just have to correct for cable delays with alpha timing
    # num_events = len(file_traces["adc_b2_ch1"])

    # # event loop
    # peak_sample_time_list = []
    # twice_checked_alpha_event_index_list = []
    # for i in range(num_events):
    #     corrected_waveforms_per_event = []
    #     # waveform loop to get the i_th waveform for each PMT
    #     for key in file_traces.keys():
    #         if ("b5" in key) or (key in IRRELEVANT_CHANNELS):
    #             continue
    #         # print(key, type(file_traces[key][i]), len(file_traces[key][i]))
    #         # time.sleep(5)
    #         board_num = int(key[5])
    #         uncorrected_waveform = file_traces[key][i]
    #         corrected_waveforms_per_event.append(
    #             waveform_daisy_correction(uncorrected_waveform, board_num)
    #         )
    #     summed_waveform = np.sum(corrected_waveforms_per_event, axis=0)
    #     peak_sample_time_ns = np.argmin(summed_waveform) * 2  # converts ADU to mV
    #     peak_sample_time_list.append(peak_sample_time_ns)
    #     if 550 < peak_sample_time_ns < 750 and i in alpha_PMT_events:
    #         twice_checked_alpha_event_index_list.append(i)

    # print(len(twice_checked_alpha_event_index_list), twice_checked_alpha_event_index_list)

    # # Let's do it again!
    # # waveforms_of_interest = []  # store as lists of [event_num, key, waveform]
    # # num_alpha_events = len(twice_checked_alpha_event_index_list)
    # channel_delays = []
    # for event_i in twice_checked_alpha_event_index_list:
    #     event_channel_delays_ns_dict = {}
    #     alpha_hit_time_i = np.argmin(\
    #     waveform_daisy_correction(file_traces["adc_b4_ch12"][event_i], 4)) * 2 # daisy correction, finds time of pulse, converts alpha hit time in ns
    #     event_channel_delays_ns_dict["adc_b4_ch12"] = alpha_hit_time_i

    #     # waveform loop to get the i_th waveform for each PMT
    #     for key in file_traces.keys():
    #         if ("b5" in key) or (key in IRRELEVANT_CHANNELS):
    #             continue
    #         board_num = int(key[5])
    #         uncorrected_waveform = file_traces[key][event_i]
    #         pmt_hit_time = np.argmin(waveform_daisy_correction(uncorrected_waveform, board_num)) * 2
    #         event_channel_delays_ns_dict[key] = pmt_hit_time - alpha_hit_time_i - CHANNELS_AND_TIMES_NS_DICT[key]

    #     channel_delays.append(event_channel_delays_ns_dict)

    """AHHHH"""

    # billy = get_channel_delays(file_traces)
    # print(len(billy))
    # for i in range(len(billy)):
    #     print(billy[i]["adc_b2_ch10"])  # 10.463753125167594

    #####
    # # MAKE MATPLOT
    # PATH = "/home/dcolson/my_histograms"
    # filename = f"histogram_test_PMT_differences.pdf"
    # output_path = os.path.join(PATH, filename)

    # # multiple plots

    # # Create the histogram
    # plt.figure(figsize=(8, 6))
    # for elem in waveforms_of_interest:
    #     if elem[0] == 43:
    #         plt.plot(elem[2], label=elem[1])
    # # plt.title(f"Waveform from adc_b5_ch34, event {i}")
    # plt.title(f"Waveforms for alpha event {43}")
    # plt.xlabel("ADU")
    # plt.ylabel("Amplitude")
    # # plt.yscale("log")

    # plt.xlim(325, 425)
    # plt.legend()
    # # plt.xticks(np.arange(0, 1001, 50))

    # # Save the figure
    # plt.savefig(output_path)
    # plt.close()

    #####
    # # Create the histogram
    # plt.figure(figsize=(8, 6))
    # plt.hist(peak_sample_time_list, bins=175, edgecolor="black")
    # plt.title("Peak Maximum Timing")
    # plt.xlabel("time in ns")
    # plt.ylabel("Frequency")
    # plt.yscale("log")

    # plt.xlim(0, 1000)
    # plt.xticks(np.arange(0, 1001, 50))

    # # Save the figure
    # plt.savefig(output_path)
    # plt.close()

    # Okay so let's write a function to get just our alpha events
    # Recall majority,alpha, and top paddles are peaks in order from left to right

    # Let's do a rough estimate that it's between 550,750 ns

    # Notes: need to implement is_pulse to get_channel_delays. that should be easy, so gonna start on 1D CNN
