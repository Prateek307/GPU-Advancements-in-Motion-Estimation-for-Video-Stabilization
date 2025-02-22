>**Note:**
>This project is built as a part of the SRIB-PRISM Program at KLE Technological University.

# GPU Advancements in Motion Estimation for Video Stabilization

This project aims to leverage advancements in GPU technology to enhance motion estimation techniques used in video stabilization algorithms like Optical Flow. The goal is to develop a Coarse-to-Fine Lucas Kanade Optical Flow algorithm that utilizes GPU's parallel processing capabilities. By doing so, we aim to significantly improve the performance of Motion Estimation part of a Video Stabilization algorithm, making it faster and more reliable.

## System Architecture
<img src="https://media.github.ecodesamsung.com/user/26797/files/9d0626a5-6d27-42e2-ab5a-0f4a24208728" width="800">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


## Tools used
<p align="left">
   <img src="https://media.github.ecodesamsung.com/user/26797/files/b26176fa-f795-43d1-aa17-3a46e842356d" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://media.github.ecodesamsung.com/user/26797/files/7dd43b1c-9349-4860-9b6f-db30ed4f36dc" height="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>

- **PyOpenCL**: provides pythonic access to the OpenCL parallel computation API.
- **NumPy**:  to perform a wide variety of mathematical operations on arrays.

## Languages used

- Python
- OpenCL C

## Installation details
>**Note:**
>Your device needs to support OpenCL.

1. - Fork the [repo](https://github.ecodesamsung.com/SRIB-PRISM/KLE_23VIS45_GPU_Advancements_in_Motion_Estimation_for_Video_Stabilization)
   - Clone the repo to your local machine `git clone https://github.ecodesamsung.com/SRIB-PRISM/KLE_23VIS45_GPU_Advancements_in_Motion_Estimation_for_Video_Stabilization`
   - Change current directory `cd KLE_23VIS45_GPU_Advancements_in_Motion_Estimation_for_Video_Stabilization`
2. Install latest version of [Python](https://www.python.org/) and create a virtual environment:
```bash
python -m venv venv
./venv/Scripts/activate
```

3. Install all dependencies:
```bash
pip install -r requirements.txt
```

4. Running the `main.py` file:  
**Command-Line Arguments**
The script supports the following command-line arguments:

- `--use`: Specify whether to use CPU or GPU for processing.
  - Choices: `CPU`, `GPU`
  - Default: `CPU`

- `--visualize`: Specify whether to visualize the optical flow.
  - Choices: `1` (True), `0` (False)
  - Default: `0`

**Example 1**: Using CPU and No Visualization
```bash
python main.py --use CPU --visualize 0
```

**Example 2**: Using GPU and Visualization
```bash
python main.py --use GPU --visualize 1
```
