import numpy as np
import nibabel as nib

def main():
    input_file = "data/FLARE-Task2-LaptopSeg/train_pseudo_label/imagesTr/Case_00001_0000.nii.gz"
    
    # Load the NIfTI file
    img = nib.load(str(input_file), mmap=True)

    # Print header metadata
    header = img.header
    print("Header metadata:")
    for key, value in header.items():
        print(f"{key}: {value}")

    # Get image data and shape
    data_np = img.get_fdata(dtype=np.float32)
    print(f"Image shape: {data_np.shape}\n")

    # Print affine transform
    print("Affine:")
    print(img.affine, "\n")

    # Native data type in storage?
    print(f"Data type in storage: {img.get_data_dtype()}\n")

    

if __name__ == "__main__":
    main()