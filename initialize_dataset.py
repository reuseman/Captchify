import managers.dataset_manager as dmng


def remove_elements_with_duplicates(list_duplicates):
    set_duplicates = set(list_duplicates)
    old_set = set_duplicates.copy()
    for element in set_duplicates:
        counter = 0
        for i in range(0, len(list_duplicates)):
            if element == list_duplicates[i]:
                counter += 1
        if counter >= 2:
            old_set.remove(element)
    return old_set


def main():
    dmng.initialize()
    # files = listdir(Commons.original_dataset_path)
    # for i in range(0, len(files)):
    #     if files[i].endswith(".jpg"):
    #         files[i] = files[i][0:-4]
    # print(remove_elements_with_duplicates(files))


if __name__ == "__main__":
    main()
