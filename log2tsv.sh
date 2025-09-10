#!/bin/bash

# Check if a log file is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <log_file>"
    exit 1
fi

# Assign the input log file from the command-line argument
log_file="$1"

# Define output TSV file
output_file="summary.tsv"

# Write header to the TSV file
echo -e "Timestep\tCFL\tt\tdt\tElapsedStep\tP_iter\tUVW_iter\tPB_iter\tBxyz_iter" > "$output_file"

# Extract relevant lines, parse the required fields, and append to TSV
awk '
{
    # Get information about individual field solves within timestep
    if (match($0, /P\s+: iter ([0-9]+)/, p_iter)) { P_iter = p_iter[1]; }
    if (match($0, /UVW\s+: iter ([0-9]+)/, uvw_iter)) { UVW_iter = uvw_iter[1]; }
    if (match($0, /PB\s+: iter ([0-9]+)/, pb_iter)) { PB_iter = pb_iter[1]; }
    if (match($0, /Bxyz\s+: iter ([0-9]+)/, bxyz_iter)) { Bxyz_iter = bxyz_iter[1]; }

    # Print on required step= lines (specific key to ignore second step= lines)
    if ((match($0, /step= ([0-9]+)/, step)) && (match($0, /dt=([0-9.e+-]+)/, dt))) {

        # Store new timestep values
        timestep = step[1];
        dt_value = dt[1];
        match($0, /t= ([0-9.e+-]+)/, time); t = time[1];
        match($0, /C= ([0-9.]+)/, c); cfl = c[1];
        match($0, /elapsedStep= ([0-9.e+-]+)/, elapsed); elapsed_step = elapsed[1];

        if (timestep) {
            # Print the previous timestep data before overwriting it with the new one
            print timestep "\t" cfl "\t" t "\t" dt_value "\t" elapsed_step "\t" \
                  P_iter "\t" \
                  UVW_iter "\t" \
                  PB_iter "\t" \
                  Bxyz_iter;
        }
        # Reset iteration counts for this new timestep
        P_iter = UVW_iter = PB_iter = Bxyz_iter = "";
    }
}' "$log_file" >> "$output_file"

echo "Data has been extracted to $output_file"
