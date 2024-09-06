#include <stdio.h>

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

extern const ei_impulse_t impulse;

// custom function to display the results
static void display_custom_results(ei_impulse_result_t* result, const ei_impulse_t* impulse);
// get_signal declaration inserted here

// raw features array inserted here

int main(int argc, char **argv) {

    signal_t signal;            // Wrapper for raw input buffer
    ei_impulse_result_t result; // Used to store inference output
    EI_IMPULSE_ERROR res;       // Return code from inference

// process_impulse inserted here

    return 0;
}

// callback functions inserted here

// callback functions inserted here

static void display_custom_results(ei_impulse_result_t* result, const ei_impulse_t *impulse)
{
    printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
            result->timing.dsp,
            result->timing.classification,
            result->timing.anomaly);

    // Print the prediction results (object detection)

    printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < impulse->object_detection_count; i++) {
        ei_impulse_result_bounding_box_t bb = result->bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }

    // Print the prediction results (classification)

    printf("Predictions:\r\n");
    for (uint16_t i = 0; i < impulse->label_count; i++) {
        printf("  %s: ", impulse->categories[i]);
        printf("%.5f\r\n", result->classification[i].value);
    }


    // Print anomaly result (if it exists)
    
        printf("Anomaly prediction: %.3f\r\n", result->anomaly);
    
#if EI_CLASSIFIER_HAS_VISUAL_ANOMALY
    // print visual anomaly
    if (impulse->has_anomaly == 3) {
        printf("Visual anomalies:\r\n");
        for (uint32_t i = 0; i < result->visual_ad_count; i++) {
            ei_impulse_result_bounding_box_t bb = result->visual_ad_grid_cells[i];
            if (bb.value == 0) {
                continue;
            }
            printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                    bb.label,
                    bb.value,
                    bb.x,
                    bb.y,
                    bb.width,
                    bb.height);
        }
        printf("Visual anomaly values: Mean : %.3f Max : %.3f\r\n",
        result->visual_ad_result.mean_value, result->visual_ad_result.max_value);
    }
#endif
}