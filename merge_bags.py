import rosbag

# Define your input files and output file
input_bags = ['./data/data_0.bag', './data/data_1.bag', './data/data_2.bag', './data/data_3.bag', './data/data_4.bag']
output_bag_name = './data/merged_output.bag'

with rosbag.Bag(output_bag_name, 'w') as outbag:
    for bag_file in input_bags:
        print(f"Processing {bag_file}...")
        with rosbag.Bag(bag_file) as inbag:
            for topic, msg, t in inbag.read_messages():
                outbag.write(topic, msg, t)
                
print("Done! Your bags are now one.")