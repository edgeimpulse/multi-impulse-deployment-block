anomaly_types = {
    "EI_ANOMALY_TYPE_UNKNOWN": 0,
    "EI_ANOMALY_TYPE_KMEANS": 1,
    "EI_ANOMALY_TYPE_GMM": 2,
    "EI_ANOMALY_TYPE_VISUAL_GMM": 3
}

object_detection_types = {
    "EI_CLASSIFIER_LAST_LAYER_UNKNOWN": -1,
    "EI_CLASSIFIER_LAST_LAYER_SSD": 1,
    "EI_CLASSIFIER_LAST_LAYER_FOMO": 2,
    "EI_CLASSIFIER_LAST_LAYER_YOLOV5": 3,
    "EI_CLASSIFIER_LAST_LAYER_YOLOX": 4,
    "EI_CLASSIFIER_LAST_LAYER_YOLOV5_V5_DRPAI": 5,
    "EI_CLASSIFIER_LAST_LAYER_YOLOV7": 6,
    "EI_CLASSIFIER_LAST_LAYER_TAO_RETINANET": 7,
    "EI_CLASSIFIER_LAST_LAYER_TAO_SSD": 8,
    "EI_CLASSIFIER_LAST_LAYER_TAO_YOLOV3": 9,
    "EI_CLASSIFIER_LAST_LAYER_TAO_YOLOV4": 10,
    "EI_CLASSIFIER_LAST_LAYER_YOLOV2": 11
}

# Function to add #define
def insert_define_statement(file_path, define_statement):
    print(f"Inserting {define_statement} into {file_path}")
    try:
        with open(file_path, 'r') as file:
            file_content = file.readlines()

        include_idx = None
        define_idx = None

        # Find the last #include and the first valid #define
        for i, line in enumerate(file_content):
            if line.startswith('#include'):
                include_idx = i

            # Check if this is a #define not immediately following an #ifndef
            if line.startswith('#define') and (i == 0 or not file_content[i - 1].strip().startswith('#ifndef')):
                define_idx = i
                break

        if include_idx is None:
            raise ValueError("Could not find any #include lines")

        if include_idx is not None and define_idx is not None:
            file_content.insert(include_idx + 1, define_statement + '\n')

        with open(file_path, 'w') as file:
            file.writelines(file_content)

        print(f"Inserted {define_statement} into {file_path}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Insert lines after a specific line in a file
def insert_after_line(file_path, search_line, lines_to_insert):
    print(f"Inserting lines into {file_path} after {search_line}")
    try:
        with open(file_path, 'r') as file:
            file_content = file.readlines()

        insert_idx = None
        for i, line in enumerate(file_content):
            if search_line in line:
                insert_idx = i + 1
                break

        if insert_idx is None:
            raise ValueError(f"Could not find the line: {search_line}")

        for line in reversed(lines_to_insert):
            file_content.insert(insert_idx, line + '\n')

        with open(file_path, 'w') as file:
            file.writelines(file_content)

        print(f"Lines inserted into {file_path}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Replace a line in a file with another
def replace_line(file_path, search_line, replacement_line):
    print(f"Replacing line in {file_path}: {search_line}")
    try:
        with open(file_path, 'r') as file:
            file_content = file.readlines()

        file_content = [line if search_line not in line else replacement_line + '\n' for line in file_content]

        with open(file_path, 'w') as file:
            file.writelines(file_content)

        print(f"Replaced line in {file_path}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Remove a line entirely from a file
def remove_line(file_path, search_string):
    print(f"Removing line from {file_path} containing {search_string}")
    try:
        with open(file_path, 'r') as file:
            file_content = file.readlines()

        file_content = [line for line in file_content if search_string not in line]

        with open(file_path, 'w') as file:
            file.writelines(file_content)

        print(f"Removed line containing {search_string} from {file_path}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")