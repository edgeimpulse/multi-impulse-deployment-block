import os, sys, argparse, json, tempfile, re, shutil
from zipfile import ZipFile
from EIDownload import EIDownload

parser = argparse.ArgumentParser(description='Multi-impulse transformation block')
parser.add_argument('--api-keys', type=str, help='List of API Keys', required=False)
parser.add_argument('--projects', type=str, help='List of project IDs separated by a comma', required=False)
parser.add_argument('--tmp-directory', type=str, required=False)
parser.add_argument('--out-directory', type=str, default='/home/output', required=False)
parser.add_argument("--float32", action="store_true", help="Use float32 model")
parser.add_argument("--force-build", action="store_true", help="Force build libraries, no cache")
parser.add_argument("--full-tflite", action="store_true", help="Use full tflite compiler and NOT EON compiler")

args, unknown = parser.parse_known_args()

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

        if args.full_tflite:
            eon = False
        else:
            eon = True

        zipfile_path = dzip.download_model(download_path, eon = eon, quantized = quantized, force_build = args.force_build)

        with ZipFile(zipfile_path, 'r') as zObject:
            zObject.extractall(download_path)
        os.remove(zipfile_path)

else:
    project_ids = args.projects.split(',')
    tmpdir = args.tmp_directory


## GENERIC FUNCTIONS TO EDIT FILES

# Generic function to add suffix to search patterns in a file
def edit_file(file_path, patterns, suffix):
    print("Editing " + file_path)
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
            print("pattern: " + pattern)
            file_content = re.sub(pattern, add_suffix, file_content)

        with open(file_path, 'w') as file:
            file.write(file_content)

        print(f"{file_path} edited")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

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

        print("Portion copied and inserted successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}")

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

        print("Merge model_ops done")

    except FileNotFoundError as e:
        print(f"Error: {e}")

# Function to check for YOLOV5
def check_for_yolov5(file_path):
    modelHasYOLOV5 = False
    print(f"Checking for YOLOV5 in {file_path}")
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            if "YOLOV5" in file_content:
                modelHasYOLOV5 = True
                print(f"YOLOV5 found")
            else:
                print(f"YOLOV5 not found")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    return modelHasYOLOV5

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

# Merge the tflite_full.h header files to include all projects
def merge_tflite_full(tmpdir, project_ids, include_pattern):
    all_includes = set()
    
    for p in project_ids:
        suffix = "_" + p

        f = os.path.join(tmpdir, p, "edge-impulse-sdk/classifier/inferencing_engines/tflite_full.h")
        print(f"Editing file: {f}")

        with open(f, 'r') as file:
            file_content = file.readlines()

        regex = re.compile(include_pattern)
        for line in file_content:
            if regex.search(line):
                new_include_line = regex.sub(r'#include "tflite-model/trained_model_ops_define' + suffix + '.h"', line).strip()
                all_includes.add(new_include_line)

    f = os.path.join(tmpdir, project_ids[0], "edge-impulse-sdk/classifier/inferencing_engines/tflite_full.h")

    with open(f, 'r') as file:
        file_content = file.readlines()

    last_include_idx = None
    for i, line in enumerate(file_content):
        if re.match(include_pattern, line):
            last_include_idx = i

    if last_include_idx is not None:
        insertion_point = last_include_idx + 1
        file_content[insertion_point:insertion_point] = [include + '\n' for include in all_includes]

        with open(f, 'w') as file:
            file.writelines(file_content)

    with open(f, 'r') as file:
        file_content = file.readlines()

    regex_unlabeled = re.compile(r'#include\s+"tflite-model/trained_model_ops_define\.h"')
    file_content = [line for line in file_content if not regex_unlabeled.match(line)]

    with open(f, 'w') as file:
        file.writelines(file_content)

## EDITING FILES

# Use first project as target dir
target_dir = os.path.join(tmpdir, project_ids[0])
include_lines = []

for p in project_ids:

    # suffix added to different functions and variables
    suffix = "_" + p
    print(f"Processing Project{str(suffix)}")

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


    # 2. Edit model_variables.h
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

    # 3. Check if full tflite and make additional required modifications
    if args.full_tflite:
        # Insert #define EI_CLASSIFIER_USE_FULL_TFLITE 1 in model_metadata.h
        f = os.path.join(tmpdir, p, "model-parameters/model_metadata.h")
        insert_define_statement(f, "#define EI_CLASSIFIER_USE_FULL_TFLITE 1")

        # Remove deprecated API reference in tflite_full.h
        f = os.path.join(tmpdir, p, "edge-impulse-sdk/classifier/inferencing_engines/tflite_full.h")
        str_rm = 'tree_ensemble_classifier'
        remove_line(f, str_rm)

        # Edit Zephyr cmake to use full tflite
        str_ins = 'set(EI_SDK_FOLDER ../../)'
        lines_to_insert = [
            'set(PROJECT_ROOT_FOLDER ../../..)',
            'set(FULL_TFLITE_FOLDER ${PROJECT_ROOT_FOLDER}/tensorflow-lite)'
        ]
        str_rep = 'LIST(APPEND EI_SOURCE_FILES "${EI_SDK_FOLDER}/tensorflow/lite/c/common.c")'
        replacement_line = 'LIST(APPEND EI_SOURCE_FILES "${FULL_TFLITE_FOLDER}/tensorflow/lite/c/common.cc")'
        f = os.path.join(tmpdir, p, "edge-impulse-sdk/cmake/zephyr/CMakeLists.txt")
        insert_after_line(f, str_ins, lines_to_insert)
        replace_line(f, str_rep, replacement_line)

    # 4. Check if model has YOLOV5 and add suffix to ei_fill_result_struct.h inside #ifdef EI_HAS_YOLOV5
    f = os.path.join(tmpdir, p, "model-parameters/model_variables.h")
    if check_for_yolov5(f):
        f = os.path.join(tmpdir, p, "edge-impulse-sdk/classifier/ei_fill_result_struct.h")
        patterns = [
            "ei_classifier_inferencing_categories"
        ]
        edit_file(f, patterns, suffix)

    # 5. Merge model_variables.h into 1st project
    if project_ids.index(p) > 0:
        merge_model_variables(f, os.path.join(target_dir, "model-parameters/model_variables.h"))

    # 6. Save intersection of trained_model_ops_define.h files
    if project_ids.index(p) > 0:
        f = os.path.join(tmpdir, p, "tflite-model/trained_model_ops_define.h")
        f2 = os.path.join(target_dir, "tflite-model/trained_model_ops_define.h")
        merge_model_ops(f, f2)

# If full tflite was used then merge the header files
if args.full_tflite:
    merge_tflite_full(tmpdir, project_ids, r'#include\s+"tflite-model/trained_model_ops_define\.h"')

# 7. Copy template files to tmpdir
shutil.copytree('templates', target_dir, dirs_exist_ok=True)

# 8 Get sample code to customize main.cpp

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

# 9. Insert custom code in main.cpp
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

print("Editing main.cpp")
with open(os.path.join(target_dir, 'source/main.cpp'), 'w') as file1:
    file1.writelines(main_template)
print("main.cpp edited")

print("Merging done!")

# 10. Create archive
shutil.make_archive(os.path.join(args.out_directory, 'deploy'), 'zip', target_dir)
