import pickle

# Original and output pickle file paths
infile_name = "/media/disk_o/my_pickles/07_29_25_disk_d_phase_3_weight.pkl"
outfile_name = "/media/disk_o/my_pickles/07_29_25_disk_d_phase_3_weight_wBP.pkl"

# Load original data
with open(infile_name, "rb") as f:
    data = pickle.load(f)

# Original lists
all_events = data['all_events_for_phase']
per_events = data['per_events_for_phase']
bp_tags = data['all_bottom_paddle_tags_for_phase']

# Sanity check: Ensure all lists are the same length
assert len(all_events) == len(per_events) == len(bp_tags), "List lengths don't match!"

# Filtered lists
filtered_all_events = []
filtered_per_events = []
filtered_bp_tags = []

# Filtering loop
for i in range(len(bp_tags)):
    if i % 1000 == 0: print("finished", i)
    bp_entry = bp_tags[i]
    if 'BP' in bp_entry and bp_entry['BP']:  # Keep only if 'BP' exists and is not empty
        filtered_all_events.append(all_events[i])
        filtered_per_events.append(per_events[i])
        filtered_bp_tags.append(bp_entry)

# Save filtered data
filtered_data = {
    'all_events_for_phase': filtered_all_events,
    'per_events_for_phase': filtered_per_events,
    'all_bottom_paddle_tags_for_phase': filtered_bp_tags,
}

with open(outfile_name, "wb") as f:
    pickle.dump(filtered_data, f)

print(f"Filtered data written to {outfile_name}")

