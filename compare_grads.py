import sys
import os
import numpy as np
import jax.numpy as jnp

def compare_grads(folder_name, rtol=1e-2, atol=1e-4):
    """
    Check if pairs of files in the given folder are close enough using numpy.allclose.

    Parameters:
    - folder_name (str): Name of the folder containing the pairs of files.
    - tolerance (float): Tolerance parameter for numpy.allclose.

    Returns:
    - results (dict): A dictionary containing the file pairs and their respective closeness status.
    """

    # List all files in the folder
    files = os.listdir(folder_name)

    # Iterate through each file pair
    for file_name in files:
        if "neuron" in file_name:
            neuron_file = os.path.join(folder_name, file_name)
            cpu_file = os.path.join(folder_name, file_name.replace("neuron", "cpu"))

            # Check if corresponding cpu file exists
            if os.path.exists(cpu_file):
                # Load data from both files
                with open(neuron_file, 'rb') as fn, open(cpu_file, 'rb')as fc:
                    print(f"neuron file {neuron_file} cpu_file {cpu_file}")
                    try:
                        while True:
                            print("inside loop")
                            neuron_data = jnp.load(fn, allow_pickle=True)
                            # print(neuron_data)
                            # trn_data = np.load(trn_file)
                            cpu_data = jnp.load(fc, allow_pickle=True)

                            # Check if data from both files are close enough
                            close_enough = jnp.allclose(neuron_data, cpu_data, rtol=rtol, atol=atol)

                            print(f"{neuron_file} {cpu_file} layer allclose check: {close_enough}")

                            print(neuron_data - cpu_data)
                    except EOFError as e:
                        print(e)

if __name__ == "__main__":
    grads_folder = sys.argv[1]
    compare_grads(grads_folder)
