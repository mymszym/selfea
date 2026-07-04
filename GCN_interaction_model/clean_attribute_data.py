"""
Reading attribute data from FBYG15K format
Format: entity <attr1> <attr2> <attr3> ... (tab separated)
"""
import os
from Param import *
from utils import fixed


def read_att_data(data_path):
    """
    load attribute data from FBYG15K format.
    Format: entity <attr1> <attr2> <attr3> ... (tab separated)
    Returns: list of (entity, attribute_name, attribute_name, 'string') tuples
    """
    print("loading attribute file from: ", data_path)
    att_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            e = parts[0].strip()
            # Each remaining part is an attribute name wrapped in <>
            for attr in parts[1:]:
                attr = attr.strip()
                if attr.startswith('<') and attr.endswith('>'):
                    attr = attr[1:-1]  # Remove < >
                # Extract attribute name from URI
                if '/' in attr:
                    attr_name = attr.split('/')[-1]
                else:
                    attr_name = attr
                # Use attribute name as both attribute and value
                att_data.append((e, attr_name, attr_name, 'string'))
    return att_data


def remove_duplicate_attributes(att_data):
    """
    Remove duplicate (entity, attribute) pairs
    """
    seen = set()
    unique_data = []
    for e, a, l, l_type in att_data:
        if (e, a) not in seen:
            seen.add((e, a))
            unique_data.append((e, a, l, l_type))
    return unique_data


def save_attribute_data(data, file_path):
    """
    save attribute data to file
    Format: entity <tab> attribute <tab> value <tab> type
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for e, a, l, l_type in data:
            f.write(f"{e}\t{a}\t{l}\t{l_type}\n")


def main():
    fixed(SEED_NUM)
    print("----------------clean attribute data--------------------")
    print("Start loading attribute data from FBYG15K format")

    # load attribute data
    att_data_1 = read_att_data(DATA_PATH + 'training_attrs_1')
    att_data_2 = read_att_data(DATA_PATH + 'training_attrs_2')

    # remove duplicates
    att_data_1 = remove_duplicate_attributes(att_data_1)
    att_data_2 = remove_duplicate_attributes(att_data_2)

    print("KG1 attribute records num: {}".format(len(att_data_1)))
    print("KG2 attribute records num: {}".format(len(att_data_2)))

    # save path
    new_attribute_file_1 = DATA_PATH + 'new_att_triples_1'
    new_attribute_file_2 = DATA_PATH + 'new_att_triples_2'

    # save
    save_attribute_data(att_data_1, new_attribute_file_1)
    save_attribute_data(att_data_2, new_attribute_file_2)

    print("save attribute data to: ", new_attribute_file_1)
    print("save attribute data to: ", new_attribute_file_2)


if __name__ == '__main__':
    main()
