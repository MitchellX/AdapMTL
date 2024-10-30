import csv

# list of dictionaries to store the log values
path_segment_semantic = "logs/task_differences/segment_semantic_depth_zbuffer_differences_2023-03-21.txt"
path_normal = "logs/task_differences/normal_depth_zbuffer_differences_2023-03-21.txt"
path_depth = "logs/task_differences/depth_zbuffer_depth_zbuffer_differences_2023-03-21.txt"
# path_segment_semantic = "logs/task_differences/resnet34_segment_semantic_differences_2023-03-09.txt"
# path_normal = "logs/task_differences/resnet34_normal_differences_2023-03-09.txt"
# path_depth = "logs/task_differences/resnet34_depth_zbuffer_differences_2023-03-09.txt"
test_score_segment_semantic = []
test_score_normal = []
test_score_depth = []

def pickup_values(path, test_score_segment_semantic): 
    # parse the log and extract the values
    with open(path, 'r') as log_file:
        for line in log_file:
            if line.__contains__('test score: '):
                test_score_segment_semantic.append(line.strip().split('test score: ')[1])

pickup_values(path_segment_semantic, test_score_segment_semantic)
pickup_values(path_normal, test_score_normal)
pickup_values(path_depth, test_score_depth)


# # write the data to a CSV file
with open('task_head_sensitivity.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["task", "0.9", "0.93", "0.95", "0.97", "0.99"])
    writer.writerow(["segment"] + test_score_segment_semantic)
    writer.writerow(["normal"] + test_score_normal)
    writer.writerow(["depth"] + test_score_depth)
