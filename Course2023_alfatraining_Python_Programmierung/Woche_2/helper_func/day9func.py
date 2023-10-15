# Python, using Anaconda environment
# Week 2, Day 9

def split_two(input_list):
    list_length = len(input_list)
    half_length = (list_length + 1) // 2

    # split the list into 2 lists
    if list_length > 2:
        left_list = input_list[:half_length]
        right_list = input_list[half_length:]
        # print(left_list)
        # print(right_list)
        return [left_list, right_list]
    else:
        return [input_list]

def split_em(*input_lists):
    # print(input_lists, type(input_lists))
    output_list = []

    for mylist in input_lists:
        # print(mylist, type(mylist))
        temp_lists = split_two(mylist)
        # print(temp_lists)
        for temp_list in temp_lists:
            output_list.append(temp_list)
    # print(output_list, "size:  ", len(output_list))

    # check size
    need_split = False
    for mylist in output_list:
        if len(mylist) > 2:
            need_split = True
            # print(mylist)
            break
        else:
            need_split = False

    if need_split:
        return split_em(*output_list)
    else:
        return output_list

def swap_two_element(input_list):
    # print(input_list)
    if len(input_list) == 2:
        # swap elements for sorting
        if input_list[0] > input_list[1]:
            input_list[0], input_list[1] = input_list[1], input_list[0]
    return input_list


def merge_em(input_list):
    print(input_list, type(input_list))
    list_length = len(input_list)

    idx = 0
    out_list = []
    while idx < list_length:
        # get 2 lists
        left_list = input_list[idx]
        right_list = []
        idx +=1
        if idx < list_length:
            right_list = input_list[idx]
            idx +=1
        # print(left_list, len(left_list))
        # print(right_list, len(right_list))

        # merge 2 lists into 1
        temp_list = []
        left_length = len(left_list)
        right_length = len(right_list)
        if right_length == 0:
            temp_list += left_list
        else:
            # sort and merge
            for left_idx, left_value in enumerate(left_list):
                for right_idx, right_value in enumerate(right_list):
                    print(left_value > right_value)
            # temp_list += left_list + right_list
            # TODO: finish this because this is wrong
        print(temp_list)

        out_list.append(temp_list)

    return out_list



def merge_sort(input_list):
    print(input_list, type(input_list))
    # split into pairs and single
    input_list = split_em(input_list)
    # print(input_list, type(input_list))

    # sort each elements
    temp_list = []
    for mylist in input_list:
        temp_list.append(swap_two_element(mylist))
    print(temp_list, type(temp_list))

    # sorting and merging elements
    temp_list = merge_em(temp_list)


    return temp_list









