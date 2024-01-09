#! /usr/bin/env python

# Summary:
#   An implementation of the Chou-Fasman algorithm
# Authors:
#   Samuel A. Rebelsky (layout of the program see:
#   http://www.cs.grinnell.edu/~rebelsky/ExBioPy/Projects/project-7.5.html)
#   Nicolas Girault


import string
import sys

import numpy as np

protein1 = 'MKIDAIVGRNSAKDIRTEERARVQLGNVVTAAALHGGIRISDQTTNSVETVVGKGESRVLIGNEYGGKGFWDNHHHHHH'
protein2 = 'MRRYEVNIVLNPNLDQSQLALEKEIIQRALENYGARVEKVAILGLRRLAYPIAKDPQGYFLWYQVEMPEDRVNDLARELRIRDNVRRVMVVKSQEPFLANA'
protein3 = 'MVGLTTLFWLGAIGMLVGTLAFAWAGRDAGSGERRYYVTLVGISGIAAVAYVVMALGVGWVPVAERTVFAPRYIDWILTTPLIVYFLGLLAGLDSREFGIVITLNTVVMLAGFAGAMVPGIERYALFGMGAVAFLGLVYYLVGPMTESASQRSSGIKSLYVRLRNLTVILWAIYPFIWLLGPPGVALLTPTVDVALIVYLDLVTKVGFGFIALDAAATLRAEHGESLAGVDTDAPAVAD'

# The Chou-Fasman table, with rows of the table indexed by amino acid name.
#   Data copied, pasted, and reformatted from 
#     http://prowl.rockefeller.edu/aainfo/chou.htm

CF = {}
# Columns are    SYM,P(a), P(b),P(turn), f(i),f(i+1), f(i+2), f(i+3)
CF['Alanine'] = ['A', 142, 83, 66, 0.06, 0.076, 0.035, 0.058]
CF['Arginine'] = ['R', 98, 93, 95, 0.070, 0.106, 0.099, 0.085]
CF['Aspartic Acid'] = ['N', 101, 54, 146, 0.147, 0.110, 0.179, 0.081]
CF['Asparagine'] = ['D', 67, 89, 156, 0.161, 0.083, 0.191, 0.091]
CF['Cysteine'] = ['C', 70, 119, 119, 0.149, 0.050, 0.117, 0.128]
CF['Glutamic Acid'] = ['E', 151, 37, 74, 0.056, 0.060, 0.077, 0.064]
CF['Glutamine'] = ['Q', 111, 110, 98, 0.074, 0.098, 0.037, 0.098]
CF['Glycine'] = ['G', 57, 75, 156, 0.102, 0.085, 0.190, 0.152]
CF['Histidine'] = ['H', 100, 87, 95, 0.140, 0.047, 0.093, 0.054]
CF['Isoleucine'] = ['I', 108, 160, 47, 0.043, 0.034, 0.013, 0.056]
CF['Leucine'] = ['L', 121, 130, 59, 0.061, 0.025, 0.036, 0.070]
CF['Lysine'] = ['K', 114, 74, 101, 0.055, 0.115, 0.072, 0.095]
CF['kLysine'] = ['k', 114, 74, 101, 0.055, 0.115, 0.072, 0.095] # 额外添加的
CF['Methionine'] = ['M', 145, 105, 60, 0.068, 0.082, 0.014, 0.055]
CF['Phenylalanine'] = ['F', 113, 138, 60, 0.059, 0.041, 0.065, 0.065]
CF['Proline'] = ['P', 57, 55, 152, 0.102, 0.301, 0.034, 0.068]
CF['Serine'] = ['S', 77, 75, 143, 0.120, 0.139, 0.125, 0.106]
CF['Threonine'] = ['T', 83, 119, 96, 0.086, 0.108, 0.065, 0.079]
CF['Tryptophan'] = ['W', 108, 137, 96, 0.077, 0.013, 0.064, 0.167]
CF['Tyrosine'] = ['Y', 69, 147, 114, 0.082, 0.065, 0.114, 0.125]
CF['Valine'] = ['V', 106, 170, 50, 0.062, 0.048, 0.028, 0.053]
CF['Jaline'] = ['J', 106, 170, 50, 0.062, 0.048, 0.028, 0.053] # 额外添加的

aa_names = ['Alanine', 'Arginine', 'Asparagine', 'Aspartic Acid',
            'Cysteine', 'Glutamic Acid', 'Glutamine', 'Glycine',
            'Histidine', 'Isoleucine', 'Leucine', 'Lysine',
            'Methionine', 'Phenylalanine', 'Proline', 'Serine',
            'Threonine', 'Tryptophan', 'Tyrosine', 'Valine', 'Jaline', 'kLysine']

Pa = {}
Pb = {}
Pturn = {}
F0 = {}
F1 = {}
F2 = {}
F3 = {}

# Convert the Chou-Fasman table above to more convenient formats
# 将上面的Chou-Fasman表格转换成更方便的格式
#    Note that for any amino acid, aa CF[aa][0] gives the abbreviation of the amino acid.给出氨基酸缩写
for aa in aa_names:
    Pa[CF[aa][0]] = CF[aa][1]
    Pb[CF[aa][0]] = CF[aa][2]
    Pturn[CF[aa][0]] = CF[aa][3]
    F0[CF[aa][0]] = CF[aa][4]
    F1[CF[aa][0]] = CF[aa][5]
    F2[CF[aa][0]] = CF[aa][6]
    F3[CF[aa][0]] = CF[aa][7]


def CF_find_alpha(seq):
    """Find all likely alpha helices in sequence.  Returns a list
       of [start,end] pairs for the alpha helices.按顺序找到所有可能的阿尔法螺旋。返回一个列表
        阿尔法螺旋的[开始，结束]对"""
    start = 0
    results = []
    # Try each window
    while (start + 6 < len(seq)):
        # Count the number of "good" amino acids (those likely to be in an alpha helix).
        # 计算“好”氨基酸的数量(那些可能在α螺旋中的氨基酸)。
        numgood = 0
        for i in range(start, start + 6):
            if (Pa[seq[i]] > 100):
                numgood = numgood + 1
        if (numgood >= 4):
            [estart, end] = CF_extend_alpha(seq, start, start + 6)
            # print "Exploring potential alpha " + str(estart) + ":" + str(end)
            # 输出有可能出现α螺旋的序列
            # if (CF_good_alpha(seq[estart:end])):
            if [estart, end] not in results:
                results.append([estart, end])
        # Go on to the next frame继续下一帧
        start = start + 1
    # That's it, we're done
    return results


def CF_extend_alpha(seq, start, end):
    """Extend a potential alpha helix sequence.
       Return the endpoints of the extended sequence.
       延伸一个潜在的阿尔法螺旋序列。返回扩展序列的端点。
    """
    # We extend the region in both directions until the average propensity for a set of four 
    # contiguous residues has Pa( ) < 100, which means we assume the helix ends there
    # 我们在两个方向上延伸该区域，直到一组四个连续残基的平均倾向Pa( ) < 100，这意味着我们假设螺旋终止于此

    # seq[end-3:end+1] is: x | x | x | END
    while (float(sum([Pa[x] for x in seq[end - 3:end + 1]])) / float(4)) > 100 and end < len(seq) - 1:
        end += 1
    # seq[start:start+4] is: START | x | x | x
    while (float(sum([Pa[x] for x in seq[start:start + 4]])) / float(4)) > 100 and start > 0:
        start -= 1

    return [start, end]


# def CF_good_alpha(subseq):
#    """Determine if a subsequence appears to be an alpha helix."""
#    sum_Pa = 0
#    for aa in subseq:
#        sum_Pa = sum_Pa + Pa[aa]
#    ave_Pa = sum_Pa/len(subseq)
#    # Criteria need to be extended
#    return (ave_Pa > 100)

def CF_find_beta(seq):
    """Find all likely beta strands in seq.  Returns a list
       of [start,end] pairs for the beta strands.
       在序列中找到所有可能的β链。返回β链的[开始，结束]对列表。"""
    start = 0
    results = []
    # Try each window
    while (start + 5 < len(seq)):
        # Count the number of "good" amino acids (those likely to be in an beta sheet).
        numgood = 0
        for i in range(start, start + 5):
            if (Pb[seq[i]] > 100):
                numgood = numgood + 1
        if (numgood >= 3):
            [estart, end] = CF_extend_beta(seq, start, start + 5)
            # print "Exploring potential alpha " + str(estart) + ":" + str(end)
            # if (CF_good_alpha(seq[estart:end])):
            if [estart, end] not in results:
                results.append([estart, end])
        # Go on to the next frame
        start = start + 1
    # That's it, we're done
    return results


def CF_extend_beta(seq, start, end):
    """Extend a potential beta helix sequence.  Return the endpoints
       of the extended sequence.
    """
    # We extend the region in both directions until the average propensity for a set of four 
    # contiguous residues has Pa( ) < 100, which means we assume the helix ends there

    # seq[end-3:end+1] is: x | x | x | END
    while (float(sum([Pb[x] for x in seq[end - 3:end + 1]])) / float(4)) > 100 and end < len(seq) - 1:
        end += 1
    # seq[start:start+4] is: START | x | x | x
    while (float(sum([Pb[x] for x in seq[start:start + 4]])) / float(4)) > 100 and start > 0:
        start -= 1
    return [start, end]


def CF_find_turns(seq):
    """Find all likely beta turns in seq.  Returns a list of positions
       which are likely to be turns.在序列中找到所有可能的beta回合。返回可能会出现的位置列表"""
    result = []
    for i in range(len(seq) - 3):
        # CONDITION 1
        c1 = F0[seq[i]] * F1[seq[i + 1]] * F2[seq[i + 2]] * F3[seq[i + 3]] > 0.000075
        # CONDITION 2
        c2 = (float(sum([Pturn[x] for x in seq[i:i + 4]])) / float(4)) > 100
        # CONDITION 3
        c3 = sum([Pturn[x] for x in seq[i:i + 4]]) > max(sum([Pa[x] for x in seq[i:i + 4]]),
                                                         sum([Pb[x] for x in seq[i:i + 4]]))
        if c1 and c2 and c3:
            result.append(i)
    return result


def region_overlap(region_a, region_b):
    """Given two regions, represented as two-element lists, determine
       if the two regions overlap.给定两个区域，表示为两个元素的列表，确定这两个区域是否重叠。
    """
    return (region_a[0] <= region_b[0] <= region_a[1]) or \
           (region_b[0] <= region_a[0] <= region_b[1])


def region_merge(region_a, region_b):
    """Given two regions, represented as two-element lists, return
       the minimum region that contains both regions.给定两个区域，表示为两个元素的列表，
返回包含这两个区域的最小区域。
    """
    return [min(region_a[0], region_b[0]), max(region_a[1], region_b[1])]


def region_intersect(region_a, region_b):
    """Given two regions, represented as two-element lists, return
       the intersection of the two regions.返回两个区域的交集。
    """
    return [max(region_a[0], region_b[0]), min(region_a[1], region_b[1])]


def region_difference(region_a, region_b):
    """Given two regions, represented as two-element lists, return
       the part of region_a which in not in region_b.
        It can be one or two regions depending on the position
        of region_b and its size.
        给定两个区域，表示为两个元素的列表，返回region_a中不在region_b中的部分。
        根据region_b的位置和大小，它可以是一个或两个区域。
    """
    # region_a start before region_b and stop before region_b
    # 区域a在区域b之前开始，在区域b之前停止
    if region_a[0] < region_b[0] and region_a[1] <= region_b[1]:
        return [[region_a[0], region_b[0] - 1]]
    # region_a start after region_b and stop after region_b
    # 区域a在区域b之后开始，在区域b之后停止
    elif region_a[0] >= region_b[0] and region_a[1] > region_b[1]:
        return [[region_b[1] + 1, region_a[1]]]
    # region_b is included in region_a => return 2 regions
    elif region_a[0] < region_b[0] and region_a[1] > region_b[1]:
        return [[region_a[0], region_b[0] - 1], [region_b[1] + 1, region_a[1]]]
    # region_a is included in region_b
    # 区域a包含在区域b中
    else:
        return []


def ChouFasman(seq):
    """Analyze seq using the Chou-Fasman algorithm and display
       the results.  A represents 'alpha helix'.  B represents 
       'beta strand'.  T represents "turn".  Space represents
       'coil structure'.
       α螺旋，β折叠， β转角，空格代表无规则卷曲（coils）
    """

    # Find probable locations of alpha helices, beta strands, and beta turns.
    alphas = CF_find_alpha(seq)
    # print "Alphas = " + str(alphas)
    betas = CF_find_beta(seq)
    # print "Betas = " + str(betas)
    turns = CF_find_turns(seq)
    # print "Turns = " + str(turns)

    # Handle overlapping regions between alpha helix and beta strands
    # 处理α螺旋和β链之间的重叠区域
    # SEE COMMENT IN MY REPORT: WHY I DONT MERGE THE ALPHA AND BETA REGIONS TOGETHER
    # 参见我的报告中的评论:为什么我DONT合并阿尔法和贝塔地区在一起
    '''# First we merge the alpha helix regions together
    x = 0
    while x < len(alphas)-1:
        if region_overlap(alphas[x],alphas[x+1]):
            alphas[x] = region_merge(alphas[x],alphas[x+1])
            alphas.pop(x+1)
        else:
          x += 1
    print "Potential alphas = " + str(alphas)
    # The same for beta strand regions
    x = 0
    while x < len(betas)-1:
        if region_overlap(betas[x],betas[x+1]):
            betas[x] = region_merge(betas[x],betas[x+1])
            betas.pop(x+1)
        else:
          x += 1
    print "Ptential betas = " + str(betas)'''

    # Then it's really messy!然后就真的乱了！
    alphas2 = []
    alphas_to_test = alphas
    betas_to_test = betas
    while len(alphas_to_test) > 0:
        alpha = alphas_to_test.pop()
        # a_shorten record if the alpha helix region has been shorten如果α螺旋区被缩短，则记录a_shorten
        a_shorten = False
        for beta in betas_to_test:
            if region_overlap(alpha, beta):
                inter = region_intersect(alpha, beta)
                # print ('Now studying overlap: ' + str(inter))
                sum_Pa = sum([Pa[seq[i]] for i in range(inter[0], inter[1] + 1)])
                sum_Pb = sum([Pb[seq[i]] for i in range(inter[0], inter[1] + 1)])

                if sum_Pa > sum_Pb:
                    # No more uncertainty on this overlap region: it will be a alpha helix
                    # 这个重叠区域不再有不确定性:它将是一个阿尔法螺旋
                    diff = region_difference(beta, alpha)
                    # print ('\tAlpha helix WIN - beta sheet region becomes: ' + str(diff))
                    for d in diff:
                        if d[1] - d[0] > 4:
                            betas_to_test.append(d)
                    betas_to_test.remove(beta)
                else:
                    # No more uncertainty on this overlap region: it will be a beta strand
                    # 这个重叠区域不再有不确定性:它将是一个β链
                    a_shorten = True
                    diff = region_difference(alpha, beta)
                    # print ('\tBeta sheet WIN - alpha helix region becomes: ' + str(diff))
                    for d in diff:
                        if d[1] - d[0] > 4:
                            alphas_to_test.append(d)
        if not a_shorten:
            alphas2.append(alpha)

    alphas = alphas2
    betas = betas_to_test

    # print('final alphas: ' + str(alphas))
    # print('final betas: ' + str(betas))
    # Build a sequence of spaces of the same length as seq.
    # 构建一个与seq长度相同的0序列。（0表示无规则卷曲）
    analysis = [0 for i in range(len(seq))]

    # Fill in the predicted alpha helices填写预测的阿尔法螺旋
    for alpha in alphas:
        for i in range(alpha[0], alpha[1]):
            analysis[i] = 1  # 1表示阿尔法螺旋
    # Fill in the predicted beta strands 填入预测的β折叠
    for beta in betas:
        for i in range(beta[0], beta[1]):
            analysis[i] = 2  # 2表示β折叠
    # Fill in the predicted beta turns 填入β转角
    for turn in turns:
        analysis[turn] = 3  # 3表示β转角

    # Turn the analysis and the sequence into strings for ease of printing
    # 将分析和序列转换成字符串，以便于打印
    # astr = str.join(analysis, '') 报错
    # sstr = str.join(seq, '')

    # 扩展成 [lenth, 1] 这样的二维数据，增加到原本的数据后面
    analysis = np.array(analysis)
    # analysis = analysis.reshape(len(analysis), 1)
    # analysis = np.tile(analysis, 10)

    return analysis


def getSequenceData(first_dir, file_name):
    # getting sequence data and label
    data, label = [], []
    path = "{}/{}.txt".format(first_dir, file_name)

    with open(path) as f:
        for each in f:
            each = each.strip()
            if each[0] == '>':
                label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
            else:
                data.append(each)

    return data, label

# 保存生成的多分类数据
def gen_CF():
    first_dir = 'dataset'

    max_length = 50  # the longest length of the peptide sequence


    # getting train data and test data
    train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train')
    test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test')


    # Converting the list collection to an array
    y_train = np.array(train_sequence_label)
    y_test = np.array(test_sequence_label)


    # 为训练数据和测试数据拼接二级结构特征
    length = 50
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    with open('cf_train_data.txt', 'w') as file:
        for i in range(len(y_train)):
            # 判断数据是否符合规范，如果不符合直接跳出，不进行处理保存
            st = train_sequence_data[i].strip()
            for j in st:
                if j not in amino_acids:
                    sign = 1
                    break
                sign = 0
            if sign == 0:
                cf_x_train = ChouFasman(train_sequence_data[i])
                cf_x_train = list(cf_x_train)
                if len(cf_x_train) >= length:
                    cf_x_train = cf_x_train[0:length-1]
                    cf_x_train.append(4)
                else:
                    for i in range(length - len(cf_x_train)):
                        cf_x_train.append(4)
                file.write(str(cf_x_train) + '\n')

    with open('cf_test_data.txt', 'w') as file:
        for i in range(len(y_test)):
            # 判断数据是否符合规范，如果不符合直接跳出，不进行处理保存
            st = test_sequence_data[i].strip()
            for j in st:
                if j not in amino_acids:
                    sign = 1
                    break
                sign = 0
            if sign == 0:
                cf_x_test = ChouFasman(test_sequence_data[i])
                cf_x_test = list(cf_x_test)
                if len(cf_x_test) >= length:
                    cf_x_test = cf_x_test[0:length-1]
                    cf_x_test.append(4)
                else:
                    for i in range(length - len(cf_x_test)):
                        cf_x_test.append(4)
                print(cf_x_test)
                file.write(str(cf_x_test) + '\n')


if __name__ == '__main__':
    gen_CF()