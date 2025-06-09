import numpy as np
from monai.transforms import ThresholdIntensity

# Create an array from 1 to 20
data = np.arange(1, 21)

# Instantiate ThresholdIntensity (array version for demonstration)
# For values above 14, only values >14 are kept, the rest are set to 0
threshold_above = ThresholdIntensity(threshold=14, above=True, cval=0)
result_above = threshold_above(data)

# For values above=False, values <14 are kept, the rest are set to 0
threshold_below = ThresholdIntensity(threshold=14, above=False, cval=0)
result_below = threshold_below(data)

print("Original data:", data)
print("After thresholding above 14:", result_above)
print("After thresholding below 14:", result_below)

