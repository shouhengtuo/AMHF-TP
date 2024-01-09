from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
import numpy as np

# peptide_smiles = Chem.MolToSmiles(Chem.MolFromFASTA("RGDfK"))
# print(peptide_smiles)


def sf_line(raw_path):
    # getting sequence data and label
    data, label = [], []
    # path = "{}/{}.txt".format(first_dir, file_name)
    with open(raw_path) as f:
        for each in f:
            each = each.strip()
            if each[0] == '>':
                label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
            else:
                data.append(each)
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0

    # 保存字符串到txt文件
    with open('Pep_train.txt', 'w') as file:
        for i in range(len(data)):
            length = len(data[i])
            st = data[i].strip()
            for j in st:
                if j not in amino_acids:
                    sign = 1
                    break
                sign = 0
            if sign == 0:
                # array_string = ' '.join(map(str, data[i]))
                peptide_smiles = ' '.join(map(str, data[i])).replace(" " ,"") + '\n'
                # peptide_smiles = Chem.MolToSmiles(Chem.MolFromFASTA(array_string)) + '\n'

                index = np.where(label[i] == 1)[0]
                if len(index) > 0:
                    label_e.append(index[0] + 1)
                # print(peptide_smiles)
                # print(label_e[i])
                file.write(peptide_smiles)
    label_e = np.array(label_e)
    # np.save('Peptrain.npy', label_e)


def get_lable(array):
    result = []
    for row in array:
        index = np.where(row == 1)[0]
        if len(index) > 0:
            result.append(index[0] + 1)

    result_array = np.array(result)

    np.save('test.npy', result_array)


if __name__ == '__main__':
    sf_line("train.txt")
    # test = np.load('Peptrain.npy')
    # a = list(test)
    # print(test)
