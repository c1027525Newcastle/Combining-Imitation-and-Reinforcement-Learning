"""
This file is used to reformat the maps from the f1tenth gym to be compatible with the learning algorithms of this project
"""

def reformat_map_centerline(file):
    actual_file = file + "_centerline.csv"

    with open(actual_file, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].replace(" ", "")  
    
    with open(actual_file, 'w') as f:
        f.writelines(lines)


def reformat_map_raceline(file):
    actual_file = file + "_raceline.csv"

    with open(actual_file, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].replace(";", ",")  

    with open(actual_file, 'w') as f:
        f.writelines(lines)


if __name__ == "__main__":
    # Change the file to the relative path of the new map that needs reformatting
    # Example: "maps/gbr"
    file = "maps/silverstone"

    reformat_map_centerline(file)
    reformat_map_raceline(file)

    print("Map reformatted")
