"""Using highly sampled csv_data from /media/disk_a/wfm 2in_2 and 3in_2 """

import matplotlib.pyplot as plt

# Path to your CSV file
csv_file_path = '/media/disk_a/wfm/2in_2/2in_2__ch1_20240604145306005.csv'

# Number of lines to skip
lines_to_skip = 14

csv_data = []

with open(csv_file_path, mode='r', encoding='utf-8') as file:
    lines = file.readlines()

# Skip the first few lines and strip newline characters
for line in lines[lines_to_skip:]:
    # Split CSV line into fields
    fields = line.strip().split(',')
    csv_data.append(fields)

# Print extracted csv_data
for row in csv_data[:5]:
    print(row)

# Example csv_data
x = [float(row[0]) for row in csv_data]
y = [float(row[1]) for row in csv_data]

# Create the plot
plt.plot(x, y, linestyle='-', color='b', label='y vs x')

# Add labels and title
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('Simple x vs y Plot')
plt.legend()

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.savefig("/media/disk_o/my_histograms/ligma.pdf")

