import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_entropy(hist):
    hist_normalized = hist / hist.sum() if hist.sum() > 0 else hist
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
    return entropy

# def split_histogram(hist, min_span=2):
#     #until min_span is reached
#     total_entropy = calculate_entropy(hist)
#     cumulative_entropy = 0
#     split_index = 0
#     for i in range(len(hist)):
#         prob = hist[i] / hist.sum()
#         cumulative_entropy += -prob * np.log2(prob + 1e-10)
#         if cumulative_entropy >= total_entropy / 2:
#             split_index = i
#             break
#     left_hist, right_hist = hist[:split_index + 1], hist[split_index + 1:]
#     return left_hist, right_hist, split_index

def recursive_histogram_split(hist, min_span=2):
    #adaptive non-zero region splits
    divisions = []
    
    def split_recursive(hist, min_level, max_level):
        span = max_level - min_level
        if span < min_span or hist.sum() == 0:
            divisions.append((hist, min_level, max_level))
            return
        
        non_zero_indices = np.nonzero(hist)[0]
        if len(non_zero_indices) == 0:
            divisions.append((hist, min_level, max_level))
            return

        start, end = non_zero_indices[0], non_zero_indices[-1]
        if end - start < min_span:
            divisions.append((hist, min_level, max_level))
            return

        adjusted_hist = hist[start:end + 1]
        split_index = (end - start) // 2 + start
        left_hist, right_hist = hist[:split_index + 1], hist[split_index + 1:]

        split_recursive(left_hist, min_level, min_level + split_index)
        split_recursive(right_hist, min_level + split_index + 1, max_level)

    split_recursive(hist, 0, len(hist) - 1)
    return divisions

def calculate_dynamic_range(sub_hist, tau=1.5):
    hist, min_level, max_level = sub_hist
    span = max_level - min_level
    entropy = calculate_entropy(hist)
    N_u = np.count_nonzero(hist)
    N_m = len(hist) - N_u
    factor = span * entropy * abs(N_u - N_m) / (N_u ** tau)
    return factor

def allocate_dynamic_range(sub_histograms, L=256, min_range=8, max_range=40):
    factors = [calculate_dynamic_range(sub_hist) for sub_hist in sub_histograms]
    factor_sum = sum(factors)
    if factor_sum > 0:
        dynamic_ranges = [(L - 1) * (factor / factor_sum) for factor in factors]
    else:
        dynamic_ranges = [L / len(sub_histograms)] * len(sub_histograms)

    dynamic_ranges = [max(min_range, min(range_i, max_range)) for range_i in dynamic_ranges]
    range_sum = sum(dynamic_ranges)
    if range_sum > 0:
        dynamic_ranges = [(L - 1) * (range_i / range_sum) for range_i in dynamic_ranges]

    return dynamic_ranges

def apply_equalization(image, sub_histograms, dynamic_ranges):
    result_image = np.zeros_like(image)
    Y_l = 0
    for (hist, min_level, max_level), range_i in zip(sub_histograms, dynamic_ranges):
        mask = (image >= min_level) & (image <= max_level)
        sub_image = image[mask]
        
        if sub_image.size > 0:
            sub_hist, bins = np.histogram(sub_image, bins=(max_level - min_level + 1), range=(min_level, max_level))
            cdf = sub_hist.cumsum()
            cdf_normalized = cdf / cdf[-1] if cdf[-1] > 0 else cdf

            Y_u = Y_l + range_i
            new_levels = Y_l + (Y_u - Y_l) * cdf_normalized

            xp = bins[:-1]
            if len(xp) != len(new_levels):
                continue

            result_image[mask] = np.interp(sub_image, xp, new_levels).astype(np.uint8)

        Y_l += range_i

    return result_image

def enhance_image(image):
    if len(image.shape) == 2:
        # Grayscale image
        hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
        sub_histograms = recursive_histogram_split(hist)
        dynamic_ranges = allocate_dynamic_range(sub_histograms)
        enhanced_image = apply_equalization(image, sub_histograms, dynamic_ranges)
        return enhanced_image
    elif len(image.shape) == 3:
        # Color image: Split channels, enhance separately, and merge
        channels = cv2.split(image)
        enhanced_channels = []
        for ch in channels:
            hist, _ = np.histogram(ch.flatten(), bins=256, range=[0, 256])
            sub_histograms = recursive_histogram_split(hist)
            dynamic_ranges = allocate_dynamic_range(sub_histograms)
            enhanced_ch = apply_equalization(ch, sub_histograms, dynamic_ranges)
            enhanced_channels.append(enhanced_ch)
        enhanced_image = cv2.merge(enhanced_channels)
        return enhanced_image
    else:
        raise ValueError("Unsupported image format.")

# Load an example image (grayscale or color)
image_path = r"C:\Users\Tuyam\Downloads\Screenshot 2024-11-09 205643.png"
image = cv2.imread(image_path)

if image is None:
    print("Image not found. Check the file path.")
else:
    enhanced_image = enhance_image(image)

    # Display original and enhanced images
    cv2.imshow("Original Image", image)
    cv2.imshow("Enhanced Image", enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Display histogram comparison for grayscale images only
    #if len(image.shape) == 2:
    plt.figure(figsize=(10, 5))
    plt.hist(image.ravel(), bins=256, color='red', alpha=0.5, label="Original Image")
    plt.hist(enhanced_image.ravel(), bins=256, color='blue', alpha=0.5, label="Enhanced Image")
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Histogram of Original and Enhanced Images")
    plt.show()
