import argparse
import nibabel as nib

def main():
    parser = argparse.ArgumentParser(description="Load a .nii.gz file, show its shape, and print all metadata")
    parser.add_argument("input_file", type=str, help="Path to the .nii.gz file")
    args = parser.parse_args()
    
    # Load the NIfTI file
    img = nib.load(args.input_file)

    # Print header metadata
    header = img.header
    print("Header metadata:")
    for key, value in header.items():
        print(f"{key}: {value}")

    # Get image data and shape
    data = img.get_fdata()
    print(f"Image shape: {data.shape}\n")

    # Print affine transform
    print("Affine:")
    print(img.affine, "\n")

    # Native data type in storage?
    print(f"Data type in storage: {img.get_data_dtype()}\n")

    

if __name__ == "__main__":
    main()