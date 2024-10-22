import os, sys, argparse, json, tempfile, re, shutil
from zipfile import ZipFile
from EIDownload import EIDownload
from utils import *
import logging

parser = argparse.ArgumentParser(description='Multi-impulse transformation block')
parser.add_argument('--api-keys', type=str, help='List of API Keys', required=False)
parser.add_argument('--projects', type=str, help='List of project IDs separated by a comma', required=False)
parser.add_argument('--tmp-directory', type=str, required=False)
parser.add_argument('--out-directory', type=str, default='/home/output', required=False)
parser.add_argument("--float32", action="store_true", help="Use float32 model")
parser.add_argument("--force-build", action="store_true", help="Force build libraries, no cache")
parser.add_argument("--engine", type=str, choices = ['eon', 'tflite'], default='eon', help="Inferencing engine to use.")

args = parser.parse_args()

logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)

# Get projects API Keys
#projectIDs = args.projects.replace(' ', '').split(',')

## DOWNLOADING LIBS

# We bypass download if we already have projects locally in a tmp directory
if not (args.projects and args.tmp_directory):

    if not args.api_keys:
        raise(Exception('--api-keys argument not set'))
    apiKeys = args.api_keys.replace(' ', '').split(',') # comma between keys

    # verify that the input file exists and create the output directory if needed
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    # Create temp directory to store zip files and manipulate files
    # tmp_directory argument used for tests
    if args.tmp_directory:
        if not os.path.exists(args.tmp_directory):
            os.makedirs(args.tmp_directory)
        tmpdir = args.tmp_directory
    else:
        tmpdir = tempfile.mkdtemp()

    project_ids = []
    # Download C++ libs and unzip
    for i in range(len(apiKeys)):
        dzip = EIDownload(api_key = apiKeys[i])
        project_ids += [str(dzip.get_project_id())]

        download_path = os.path.join(tmpdir, str(project_ids[i]))

        os.makedirs(download_path)
        if args.float32:
            quantized = False
        else:
            quantized = True

        zipfile_path = dzip.download_model(download_path, eon = (args.engine == 'eon'), quantized = quantized, force_build = args.force_build)

        with ZipFile(zipfile_path, 'r') as zObject:
            zObject.extractall(download_path)
        os.remove(zipfile_path)

else:
    project_ids = args.projects.split(',')
    tmpdir = args.tmp_directory


## GENERIC FUNCTIONS TO EDIT FILES

# Generic function to add suffix to search patterns in a file
def edit_file(file_path, patterns, suffix):
    logger.info("Editing " + file_path)
    try:
        with open(file_path, 'r') as file:
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

        with open(file_path, 'w') as file:
            file.write(file_content)

        logger.debug(f"{file_path} edited")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

def find_highest_fft_string(src_file_contents, dest_file_contents):
    fft_macros_list = [f"EI_CLASSIFIER_LOAD_FFT_{32*num}" for num in [1, 2, 4, 8, 16, 32, 64, 128]]
    logger.debug(fft_macros_list)
    src_fft_values = []
    dest_fft_values = []

    for macro in fft_macros_list:
        num1, line_num1, str1 = find_value(src_file_contents, macro)
        num2, line_num2, str2 = find_value(dest_file_contents, macro)
        src_file_contents.pop(line_num1)
        dest_file_contents.pop(line_num2)
        src_fft_values.append((num1))
        dest_fft_values.append((num2))
    logger.debug(src_fft_values, dest_fft_values)

    # find the highest value
    src_fft_value = src_fft_values.index('1')
    dest_fft_value = dest_fft_values.index('1')
    logger.debug(src_fft_value, dest_fft_value)

    logger.info(f"Highest FFT value: {fft_macros_list[max(src_fft_value, dest_fft_value)]}")
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
        dest_file_contents[line_num2] = re.sub(str1)
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
    logger.debug(num1, num2)

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
        logger.error("Error: Version mismatch, rebuild the projects with --force-rebuild")
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
        with open(src_file, 'r') as file1:
            src_file_contents = file1.readlines()
        with open(dest_file, 'r') as file2:
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

        dest_file_contents = find_highest_fft_string(src_file_contents, dest_file_contents)
        dest_file_contents = find_common_type(src_file_contents, dest_file_contents, "EI_CLASSIFIER_OBJECT_DETECTION_LAST_LAYER", object_detection_types)
        dest_file_contents = find_common_type(src_file_contents, dest_file_contents, "EI_CLASSIFIER_HAS_ANOMALY", anomaly_types)

        with open(dest_file, 'w') as file2:
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
        with open(src_file, 'r') as file1:
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
        with open(dest_file, 'r') as file2:
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
        with open(dest_file, 'w') as file2:
            file2.writelines(file2_contents)

        logger.info("Portion copied and inserted successfully!")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")

# Function to keep intersection of model_ops_define.h
def merge_model_ops(src_file, dest_file):
    try:
        with open(src_file, 'r') as file1:
            lines_file1 = [line.strip() for line in file1.readlines()]

        with open(dest_file, 'r') as file2:
            lines_file2 = [line.strip() for line in file2.readlines()]

        # Find the intersection of lines
        intersection = [line for line in lines_file1 if line in lines_file2]

        # Write the intersection back to src_file
        with open(dest_file, 'w') as file1:
            for line in intersection:
                file1.write(line + '\n')

        logger.info("Merge model_ops done")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")

## EDITING FILES

# create a target dir
target_dir = os.path.join(args.out_directory, "output")
# copy from the first project
shutil.copytree(os.path.join(tmpdir, project_ids[0]), target_dir, dirs_exist_ok=True)
include_lines = []

for p in project_ids:

    # suffix added to different functions and variables
    suffix = "_" + p
    logger.info(f"Processing Project{str(suffix)}")

    # 1. Edit compiled files in tflite-model/
    pdir = os.path.join(tmpdir, p, 'tflite-model')
    for f in os.listdir(pdir):
        if "compiled" in f:

            # List of patterns to look for and append with suffix
            # Get learn block ID pattern to add projectID as suffix
            patterns = [
                r"tflite_learn_\d+"
            ]
            edit_file(os.path.join(pdir, f), patterns, suffix)

            # Rename filenames
            new_f = f.replace("_compiled", f"{suffix}_compiled")
            os.rename(os.path.join(pdir, f), os.path.join(pdir, new_f))

            # copy to target_dir (1st project)
            if project_ids.index(p) > 0:
                shutil.copy(os.path.join(pdir, new_f), os.path.join(target_dir, 'tflite-model', new_f))

        else:
            patterns = [
                r"tflite_learn_\d+"
            ]
            edit_file(os.path.join(pdir, f), patterns, suffix)

            # Rename filenames
            name, ext = os.path.splitext(f)
            new_f = f"{name}{suffix}{ext}"
            os.rename(os.path.join(pdir, f), os.path.join(pdir, new_f))

            # copy to target_dir (1st project)
            if project_ids.index(p) > 0:
                shutil.copy(os.path.join(pdir, new_f), os.path.join(target_dir, 'tflite-model', new_f))

    # Edit model_variables.h
    f = os.path.join(tmpdir, p, "model-parameters/model_variables.h")

    # Patterns may be missing for anomaly detection blocks
    patterns = [
        r"tflite_learn_\d+",
        r"tflite_graph_\d+",
        "ei_classifier_inferencing_categories",
        r"ei_dsp_config_\d+",
        "ei_dsp_blocks",
        "ei_learning_blocks",
        r"ei_learning_block_config_\d+",
        r"ei_learning_block_\d+_inputs",
        "ei_object_detection_nms(?!_config)",
        "ei_calibration"
    ]
    edit_file(f, patterns, suffix)

    # Merge model_variables.h into 1st project
    if project_ids.index(p) > 0:
        merge_model_variables(f, os.path.join(target_dir, "model-parameters/model_variables.h"))

    # Save intersection of trained_model_ops_define.h files
    if project_ids.index(p) > 0:
        f = os.path.join(tmpdir, p, "tflite-model/trained_model_ops_define.h")
        f2 = os.path.join(target_dir, "tflite-model/trained_model_ops_define.h")
        merge_model_ops(f, f2)

    if project_ids.index(p) > 0:
        f1 = os.path.join(tmpdir, p, "model-parameters/model_metadata.h")
        f2 = os.path.join(target_dir, "model-parameters/model_metadata.h")
        merge_model_metadata(f1, f2)

# TODO: merge the resolvers
# if args.engine == 'tflite':
#     f = os.path.join(tmpdir, p, "tflite-model/tflite-resolver.h")
#     f2 = os.path.join(target_dir, "tflite-model/tflite-resolver.h")
#     merge_model_ops(f, f2)

# Copy template files to tmpdir
shutil.copytree('templates', target_dir, dirs_exist_ok=True)

# Get sample code to customize main.cpp

# Get impulses ID from model_variables.h
with open(os.path.join(target_dir, 'model-parameters/model_variables.h'), 'r') as file:
    file_content = file.read()
impulses_id_set = set(re.findall(r"impulse_(\d+)_(\d+)", file_content))
impulses_id = {}
for i in impulses_id_set:
    impulses_id[i[0]] = i[1]

get_signal_code = "\n"
raw_features_code = "\n"
run_classifier_code = "\n"
callback_function_code = "\n"
newline = "\n"

# custom code for each project
for p in project_ids:
    get_signal_code += f"static int get_signal_data_{p}(size_t offset, size_t length, float *out_ptr);{newline}"
    raw_features_code += f"static const float features_{p}[] = {{ ... }}; // copy features from project {p}{newline}"

    deploy_version = impulses_id[p]
    run_classifier_code += f"""
    // new process_impulse call for project ID {p}
    signal.total_length = impulse_{p}_{deploy_version}.dsp_input_frame_size;
    signal.get_data = &get_signal_data_{p};
    res = process_impulse(&impulse_handle_{p}_{deploy_version}, &signal, &result, false);
    printf("process_impulse for project {p} returned: %d\\r\\n", res);
    display_custom_results(&result, &impulse_{p}_{deploy_version});
    {newline}"""

    callback_function_code += f"""
static int get_signal_data_{p}(size_t offset, size_t length, float *out_ptr) {{
    for (size_t i = 0; i < length; i++) {{
        out_ptr[i] = (features_{p} + offset)[i];
    }}
    return EIDSP_OK;
}}
{newline}"""

# Insert custom code in main.cpp
with open(os.path.join(target_dir, 'source/main.cpp'), 'r') as file1:
    main_template = file1.readlines()

idx = main_template.index("// get_signal declaration inserted here\n") +1
main_template[idx:idx] = get_signal_code
idx = main_template.index("// raw features array inserted here\n") + 1
main_template[idx:idx] = raw_features_code
idx = main_template.index("// process_impulse inserted here\n") + 1
main_template[idx:idx] = run_classifier_code
idx = main_template.index("// callback functions inserted here\n") + 1
main_template[idx:idx] = callback_function_code

logger.info("Editing main.cpp")
with open(os.path.join(target_dir, 'source/main.cpp'), 'w') as file1:
    file1.writelines(main_template)
logger.info("main.cpp edited")

logger.info("Merging done!")

# Create archive
shutil.make_archive(os.path.join(args.out_directory, 'deploy'), 'zip', target_dir)
