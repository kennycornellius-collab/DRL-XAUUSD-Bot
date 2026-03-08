import csv
import sys
from datetime import datetime, timezone

def convert_timestamps(input_file, output_file):
    with open(input_file, 'r', newline='') as infile, \
         open(output_file, 'w', newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)

        for row in reader:
            if row:
                dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M")
                dt = dt.replace(tzinfo=timezone.utc)
                row[0] = str(dt)  
                writer.writerow(row)

    print(f"Done! Converted file saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        output_file = input_file.replace(".csv", "_converted.csv")
    else:
        print("Usage: python convert_timestamps.py input.csv [output.csv]")
        sys.exit(1)

    convert_timestamps(input_file, output_file)