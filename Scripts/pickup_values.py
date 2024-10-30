import csv

# list of dictionaries to store the log values
log_path = []
Backbone = []
segment_semantic = []
normal = []
depth = []
overall_sparsity = []
iters = []
test_score = []

# parse the log and extract the values
# with open('logs/resnet34_accuracy_saprsity_2023-03-30.txt', 'r') as log_file:
# with open('logs/mobilenetv2_baseline_2023-03-30.txt', 'r') as log_file:
with open('logs/mobilenetv2_overall_2023-03-30.txt', 'r') as log_file:
    for line in log_file:
        if line.__contains__('******* path: '):
            log_path.append(line)
            # log_path.append(line.strip().split('unity')[1].split('_weighted_loss')[0])
        elif line.__contains__('Backbone sparsity: '):
            Backbone.append(line.strip().split('Backbone sparsity: ')[1])
        elif line.__contains__('segment_semantic sparsity: '):
            segment_semantic.append(line.strip().split('segment_semantic sparsity: ')[1])
        elif line.__contains__('normal sparsity: '):
            normal.append(line.strip().split('normal sparsity: ')[1])
        elif line.__contains__('depth sparsity: '):
            depth.append(line.strip().split('depth sparsity: ')[1])
        elif line.__contains__('overall_sparsity: '):
            overall_sparsity.append(line.strip().split('overall_sparsity: ')[1])
        elif line.__contains__('Task segm'):
            iters.append(line.strip().split('Iter ')[1].split(' Task segm')[0])
        elif line.__contains__('test score: '):
            test_score.append(line.strip().split('test score: ')[1])

# combine the lists into a list of tuples
data = zip(log_path, iters, Backbone, segment_semantic, normal, depth, overall_sparsity, test_score)

# write the data to a CSV file
with open('avgscore_sparsity.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["weight_decay", "iters", "backbone", "segment_semantic", "normal", "depth", "overall_sparsity", "test_score"])
    for row in data:
        writer.writerow(row)