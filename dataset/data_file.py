with open('target.txt', 'r') as source_file, \
     open('target1.txt', 'w') as target1_file, open('target2.txt', 'w') as target2_file:
    source_lines = source_file.readlines()
    num_lines = len(source_lines)

    split_index = num_lines // 2

    target1_file.writelines(source_lines[:split_index])
    target2_file.writelines(source_lines[split_index:])
