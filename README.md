To set up your working directory to run the slurm job in the cluster:
1. Follow the instructions on [Conda on ARC](https://rcs.ucalgary.ca/Conda_on_ARC#Installing_Conda).
   - Make sure to follow the instructions carefully and **decline the offer to initialize**. Also, create the script `init-conda` in the `~/software` directory, as recommended.
   - Create a conda environment named "pytorch".
2. Execute the command `module avail cuda` to see available CUDA versions in the cluster.
   - Take note of the latest gpu CUDA version available as you will need it on the next step.
3. Activate your "pytorch" conda environment and run the following commands to install the required packages into it:
   - `CONDA_OVERRIDE_CUDA="AVAILABLE_CUDA_VERSION" conda install -c conda-forge pytorch=PYTORCH_VERSION python pip`
   - `CONDA_OVERRIDE_CUDA="AVAILABLE_CUDA_VERSION" conda install -c conda-forge torchvision cuda-nvcc`
     - *Note: `AVAILABLE_CUDA_VERSION` is the latest version captured on the previous step, `PYTORCH_VERSION` is the latest pytorch version compatible with the captured CUDA version. In my case, I'm using 12.1.1 and 2.1.2, respectively. For more information, see: [PyTorch on ARC](https://rcs.ucalgary.ca/PyTorch_on_ARC)*
   - `conda install -c conda-forge transformers matplotlib numpy pandas scikit-learn seaborn`
   
4. Clone this repository into any subdirectory in your personal home directory.
5. `cd` to the cloned repository and execute the SLURM file with `sbatch`, like so:
   - `sbatch group2_job.slurm `

Additional commands:
- `squeue` to check queue.
- `squeue -u {your_username}` to check queue per user.
- `scancel {job_id}` to cancel a job.
- `sinfo`  to view partition and node information.
  