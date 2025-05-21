

def expected_calibration_error(conf_list, correct_list, num_bins, order):
    binned_error = 0.0
    label_binned_error = 0.0

    total_size = len(conf_list)
    if total_size <= 0:
        return None

    step_size = 1.0 / num_bins

    for i in range(0, num_bins):
        conf_min = i * step_size
        conf_max = (i+1) * step_size

        conf_bin = conf_list[(conf_min <= conf_list) & (conf_list < conf_max)]
        correct_bin = correct_list[(conf_min <= conf_list) & (conf_list < conf_max)]

        conf_bin_size = len(conf_bin)  # S_j
        if conf_bin_size <= 0.0:
            continue

        mean_confidence = conf_bin.mean()  # C_j
        mean_accuracy = correct_bin.sum() / conf_bin_size  # A_j

        binned_error += conf_bin_size / total_size * ((mean_accuracy - mean_confidence).abs()**order)
        label_binned_error += ((conf_bin - mean_accuracy).abs()**order).sum()

    return float(binned_error**(1/order)), float((label_binned_error/total_size)**(1/order))

