import logging
import sys
import re

logging.basicConfig()
logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)

anomaly_types = {
    "EI_ANOMALY_TYPE_UNKNOWN": 0,
    "EI_ANOMALY_TYPE_KMEANS": 1,
    "EI_ANOMALY_TYPE_GMM": 2,
    "EI_ANOMALY_TYPE_VISUAL_GMM": 3
}

object_detection_types = {
    "EI_CLASSIFIER_LAST_LAYER_UNKNOWN": 0,
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
    logger.info(f"Inserting {define_statement} into {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
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

        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(file_content)

        logger.info(f"Inserted {define_statement} into {file_path}")

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

# Insert lines after a specific line in a file
def insert_after_line(file_path, search_line, lines_to_insert):
    logger.info(f"Inserting lines into {file_path} after {search_line}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
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

        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(file_content)

        logger.info(f"Lines inserted into {file_path}")

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

# Replace a line in a file with another
def replace_line(file_path, search_line, replacement_line):
    logger.info(f"Replacing line in {file_path}: {search_line}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.readlines()

        file_content = [line if search_line not in line else replacement_line + '\n' for line in file_content]

        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(file_content)

        logger.info(f"Replaced line in {file_path}")

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

# Remove a line entirely from a file
def remove_line(file_path, search_string):
    logger.info(f"Removing line from {file_path} containing {search_string}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.readlines()

        file_content = [line for line in file_content if search_string not in line]

        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(file_content)

        logger.info(f"Removed line containing {search_string} from {file_path}")

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

## GENERIC FUNCTIONS TO EDIT FILES

# Generic function to add suffix to search patterns in a file
def edit_file(file_path, patterns, suffix):
    logger.info("Editing " + file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        # function to add suffix to search patterns
        def add_suffix(term):
            matched_text = term.group(0)
            # Check if the pattern contains 'include'
            if "include" in matched_text:
                # Special handling for include statements
                return re.sub(r'(\w+)(\.h)', rf'\1{suffix}\2', matched_text)
            else:
                # General case: simply add suffix to the matched pattern
                return matched_text + suffix

        # Search each pattern in file and call add_suffix
        for pattern in patterns:
            logger.debug("pattern: " + pattern)
            file_content = re.sub(pattern, add_suffix, file_content)

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(file_content)

        logger.debug(f"{file_path} edited")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

def find_highest_fft_string(src_file_contents, dest_file_contents):
    fft_macros_list = [f"EI_CLASSIFIER_LOAD_FFT_{32*num}" for num in [1, 2, 4, 8, 16, 32, 64, 128]]
    logger.debug(f'{fft_macros_list}')
    src_fft_values = []
    dest_fft_values = []

    for macro in fft_macros_list:
        num1, line_num1, str1 = find_value(src_file_contents, macro)
        num2, line_num2, str2 = find_value(dest_file_contents, macro)

        src_file_contents.pop(line_num1)
        dest_file_contents.pop(line_num2)
        src_fft_values.append((num1))
        dest_fft_values.append((num2))

    logger.debug(f'{src_fft_values}, {dest_fft_values}')
    logger.debug(f'{line_num1}, {line_num2}')

    # find the highest value
    try:
        # Ensure the list is reversed to get the highest value (in case multiple FFT are in one metadata file)
        src_fft_value = len(src_fft_values) - 1 - src_fft_values[::-1].index('1')
    except ValueError: # If no FFT in the impulse
        src_fft_value = 0

    try:
        dest_fft_value = len(dest_fft_values) - 1 - dest_fft_values[::-1].index('1')
    except ValueError:
        dest_fft_value = 0

    logger.debug(f'{src_fft_value}, {dest_fft_value}')

    max_fft_macro = fft_macros_list[max(src_fft_value, dest_fft_value)]

    logger.info(f"Highest FFT value: {max_fft_macro}")
    tmp = dest_file_contents[-1]
    dest_file_contents[-1] = f"#define {fft_macros_list[max(src_fft_value, dest_fft_value)]} 1\n"
    dest_file_contents.append(tmp)

    return dest_file_contents

def find_common_type(src_file_contents, dest_file_contents, macro_string, type_dict):
    src_val, line_num1, str1 = find_value(src_file_contents, macro_string)
    dest_val, line_num2, str2 = find_value(dest_file_contents, macro_string)

    logger.debug(f"Comparing {macro_string} values: {src_val}, {dest_val}")

    if src_val is None or dest_val is None:
        logger.error(f"Unknown {macro_string}, not found in one or both of the projects")
        sys.exit(1)

    # get the value, raise error if not found
    src_type = type_dict.get(src_val, None)
    dest_type = type_dict.get(dest_val, None)

    if src_type is None or dest_type is None:
        logger.error(f"Unknown type {macro_string}, not found in the type dictionary")
        sys.exit(1)

    # types match, nothing to do here
    if (src_type == dest_type):
        pass
    # if one has type and the other does not
    elif (dest_type == 0):
        dest_file_contents[line_num2] = re.sub(dest_val, src_val, str2)
    # both have types of different values
    else:
        logger.error(f"Error: {macro_string} type mismatch, can only merge projects with the same type")
        sys.exit(1)

    return dest_file_contents

def compare_values(val1, val2):
    pattern = r'#define\s+[A-Z_]+\s+(\d+)'
    num1 = re.findall(pattern, val1)
    num2 = re.findall(pattern, val2)
    print(num1, num2)

def replace_value(src_file_contents, dest_file_contents, macro_sting, choose_high_value = True):
    num1, line_num1, str1 = find_value(src_file_contents, macro_sting)
    num2, line_num2, str2 = find_value(dest_file_contents, macro_sting)
    logger.debug(f'{num1}, {num2}')

    if num1 is None:
        logger.debug(f"{macro_sting} not found in source file")
        return dest_file_contents

    if num2 is None:
        logger.debug(f"{macro_sting} not found in destination file")
        tmp = dest_file_contents[-1]
        dest_file_contents[-1] = str1
        dest_file_contents.append(tmp)
        return dest_file_contents

    # replace the value in the dest_file
    correct_num = max(int(num1), int(num2)) if choose_high_value else min(int(num1), int(num2))
    dest_file_contents[line_num2] = re.sub(num2, str(correct_num), str2)

    return dest_file_contents

def compare_version(src_file_contents, dest_file_contents):
    major_src, _, _ = find_value(src_file_contents, "EI_STUDIO_VERSION_MAJOR")
    minor_src, _, _ = find_value(src_file_contents, "EI_STUDIO_VERSION_MINOR")
    patch_src, _, _ = find_value(src_file_contents, "EI_STUDIO_VERSION_PATCH")

    major_dest, _, _ = find_value(dest_file_contents, "EI_STUDIO_VERSION_MAJOR")
    minor_dest, _, _ = find_value(dest_file_contents, "EI_STUDIO_VERSION_MINOR")
    patch_dest, _, _ = find_value(dest_file_contents, "EI_STUDIO_VERSION_PATCH")

    if major_src != major_dest or minor_src != minor_dest or patch_src != patch_dest:
        logger.error("Error: Version mismatch, rebuild the projects with --force-build")
        logger.error(f"Source version: {major_src}.{minor_src}.{patch_src}")
        logger.error(f"Destination version: {major_dest}.{minor_dest}.{patch_dest}")
        sys.exit(1)

def find_value(file_content, macro_string):
    for i, line in enumerate(file_content):
        if macro_string in line:
            val = re.findall(r'#define\s+[A-Z_0-9]+\s+([A-Z_0-9]+)', line)
            return val[0], i, line
    return None, None, None

def merge_model_metadata(src_file, dest_file):
    try:
        # Open the first file for reading
        with open(src_file, 'r', encoding='utf-8') as file1:
            src_file_contents = file1.readlines()
        with open(dest_file, 'r', encoding='utf-8') as file2:
            dest_file_contents = file2.readlines()

        compare_version(src_file_contents, dest_file_contents)

        dest_file_contents = replace_value(src_file_contents, dest_file_contents, "EI_CLASSIFIER_LABEL_COUNT")
        dest_file_contents = replace_value(src_file_contents, dest_file_contents, "EI_CLASSIFIER_HAS_VISUAL_ANOMALY")
        dest_file_contents = replace_value(src_file_contents, dest_file_contents, "EI_CLASSIFIER_SINGLE_FEATURE_INPUT", choose_high_value = False)
        dest_file_contents = replace_value(src_file_contents, dest_file_contents, "EI_CLASSIFIER_QUANTIZATION_ENABLED")
        dest_file_contents = replace_value(src_file_contents, dest_file_contents, "EI_CLASSIFIER_LOAD_IMAGE_SCALING")
        dest_file_contents = replace_value(src_file_contents, dest_file_contents, "EI_DSP_PARAMS_SPECTRAL_ANALYSIS_ANALYSIS_TYPE_FFT")
        dest_file_contents = replace_value(src_file_contents, dest_file_contents, "EI_DSP_PARAMS_SPECTRAL_ANALYSIS_ANALYSIS_TYPE_WAVELET")
        dest_file_contents = replace_value(src_file_contents, dest_file_contents, "EI_CLASSIFIER_OBJECT_DETECTION")
        dest_file_contents = replace_value(src_file_contents, dest_file_contents, "EI_CLASSIFIER_OBJECT_DETECTION_COUNT")
        dest_file_contents = replace_value(src_file_contents, dest_file_contents, "EI_CLASSIFIER_HAS_FFT_INFO")
        dest_file_contents = replace_value(src_file_contents, dest_file_contents, "EI_CLASSIFIER_NON_STANDARD_FFT_SIZES")
        fft_macros_list = [f"EI_CLASSIFIER_LOAD_FFT_{32*num}" for num in [1, 2, 4, 8, 16, 32, 64, 128]]
        # Logical OR to select each used FFT in both impulses (see edge-impulse-sdk/dsp/numpy.hpp | line 2067)
        for macro in fft_macros_list:
            dest_file_contents = replace_value(src_file_contents, dest_file_contents, macro)
        dest_file_contents = find_common_type(src_file_contents, dest_file_contents, "EI_CLASSIFIER_OBJECT_DETECTION_LAST_LAYER", object_detection_types)
        dest_file_contents = find_common_type(src_file_contents, dest_file_contents, "EI_CLASSIFIER_HAS_ANOMALY", anomaly_types)

        with open(dest_file, 'w', encoding='utf-8') as file2:
            file2.writelines("".join(dest_file_contents))

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")

# Function to merge model_variables.h
def merge_model_variables(src_file, dest_file):
    start_str = "const char* ei_classifier_inferencing_categories"
    end_str = "ei_impulse_handle_t& ei_default_impulse"
    insert_line_str = "ei_impulse_handle_t& ei_default_impulse"
    include_line_str = '#include "tflite-model/tflite_learn'
    try:
        # Open the first file for reading
        with open(src_file, 'r', encoding='utf-8') as file1:
            file1_contents = file1.readlines()

        start_line = None
        end_line = None
        include_lines = []
        for i, line in enumerate(file1_contents):
            if include_line_str in line:
                include_lines += [line]
            if start_str in line:
                start_line = i
            if end_str in line:
                end_line = i-1
                break

        if start_line is None or end_line is None:
            raise ValueError("Start or end string not found model_variables.h")

        # Open the second file for reading
        with open(dest_file, 'r', encoding='utf-8') as file2:
            file2_contents = file2.readlines()

        # Find the line number for the insertion string in the second file
        insert_line = None
        insert_include_line = None
        for i, line in enumerate(file2_contents):
            if include_line_str in line:
                insert_include_line = i
            if insert_line_str in line:
                insert_line = i
                break

        if insert_line is None:
            raise ValueError("Insertion string not found in model_variables.h")

        # Extract the portion between start and end lines from the first file
        portion_to_copy = file1_contents[start_line:end_line + 1]
        portion_to_copy[0:0] = ["\n"]
        portion_to_copy += ["\n"]

        # Insert the extracted portions into the second file at the specified lines
        file2_contents[insert_include_line:insert_include_line] = include_lines
        file2_contents[insert_line:insert_line] = portion_to_copy

        # Open the second file for writing and overwrite its contents
        with open(dest_file, 'w', encoding='utf-8') as file2:
            file2.writelines(file2_contents)

        logger.info("Portion copied and inserted successfully!")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")

# Function to keep intersection of model_ops_define.h
def merge_model_ops(src_file, dest_file):
    try:
        with open(src_file, 'r', encoding='utf-8') as file1:
            lines_file1 = [line.strip() for line in file1.readlines()]

        with open(dest_file, 'r', encoding='utf-8') as file2:
            lines_file2 = [line.strip() for line in file2.readlines()]

        # Find the intersection of lines
        intersection = [line for line in lines_file1 if line in lines_file2]

        # Write the intersection back to src_file
        with open(dest_file, 'w', encoding='utf-8') as file:
            for line in intersection:
                file.write(line + '\n')

        logger.info("Merge model_ops done")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")

def merge_tflite_resolver(src_file, dest_file):
    try:
        with open(src_file, 'r', encoding='utf-8') as file1:
            lines_file1 = [line.strip() for line in file1.readlines()]

        with open(dest_file, 'r', encoding='utf-8') as file2:
            lines_file2 = [line.strip() for line in file2.readlines()]

        # Find the union of lines
        union = []
        for line1, line2 in zip(lines_file1, lines_file2):
            if line1 in lines_file2:
                union.append(line1)
            if not line2 in union:
                union.append(line2)

        # Write the union back to src_file
        with open(dest_file, 'w', encoding='utf-8') as file:
            for line in union:
                if line.startswith('resolver.') and not line.endswith('\\'):
                    line = line + ' \\'
                file.write(line + '\n')

        logger.info("Merge tflite resolver done")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")