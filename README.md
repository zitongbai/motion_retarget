# motion_retarget
Some scripts to retarget amass dataset for humanoid robots. Most code is based on [PHC](https://github.com/ZhengyiLuo/PHC), but has been rewritten in a more readable way and adapted to my own workflow.

 The following robots are supported:

- g1_29dof_lock_waist (29 DoF, Unitree G1 with waist locked)
- Unitree H1
- Unitree G1

To add your own robot, please refer to [PHC's retargeting doc](https://github.com/ZhengyiLuo/PHC/blob/master/docs/retargeting.md). 

# Installation

1. Create a conda environment with python 3.8, install pytorch:
```bash
conda create -n retarget python=3.8
conda activate retarget
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

2. Clone the repository and install other dependencies:
```bash
git clone https://github.com/zitongbai/motion_retarget.git
# or via ssh:
# git clone git@github.com:zitongbai/motion_retarget.git
cd motion_retarget
# make sure you have activated the conda environment
pip install requirements.txt
```

3. Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/) (version 1.1.0 for Python 2.7, female/male/neutral) and [SMPLX](https://smpl-x.is.tue.mpg.de/download.php) (SMPL-X v1.1 (NPZ+PKL, 830MB)). Create folder `model/smpl` under the root directory of the repository, and unzip the downloaded files and copy them to `model/smpl` folder.  Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. The file structure should look like this:

```
├── model
    ├── smpl
        ├── SMPL_FEMALE.pkl
        ├── SMPL_MALE.pkl
        ├── SMPL_NEUTRAL.pkl
        ├── SMPLX_FEMALE.npz
        ├── SMPLX_FEMALE.pkl
        ├── SMPLX_MALE.npz
        ├── SMPLX_MALE.pkl
        ├── SMPLX_NEUTRAL.npz
        └── SMPLX_NEUTRAL.pkl
```

4. Download the AMASS dataset from [AMASS](https://amass.is.tue.mpg.de/) and extract it to somewhere else. There are many sub datasets in AMASS, you can choose to download all of them or just the ones you need. The dataset should look like this:

```
├── somewhere
    ├── AMASS
        ├──ACCAD
            ├── Female1General_c3d
            │   ├── A10 - lie to crouch_poses.npz
            │   ├── A11 - crawl forward_poses.npz
            │   └── ...
            ├── Female1Gestures_c3d
            │   ├── D1 - Urban 1_poses.npz
            │   ├── D2 - Wait 1_poses.npz
            │   └── ...
            └── ...
        ├──BMLhandball
            └── ...
        └── ...
```

Or you can create a folder and select some .npz files from the AMASS dataset, and put them in the folder. The folder structure should look like this:

```
├── somewhere
    ├── your_folder
        ├── A10 - lie to crouch_poses.npz
        ├── A11 - crawl forward_poses.npz
        ├── D1 - Urban 1_poses.npz
        └── ...
```

5. Download the meshes file for g1_29dof_lock_waist from [Unitree's official repo](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description), and put them in `resources/g1_29dof_lock_waist/meshes`. 

# Usage

Use g1_29dof_lock_waist as an example. 

1. Find the optimized shape parameters that best fit the robot:

```bash
# in the root directory of the repository
python fit_smpl_shape.py robot=g1_29dof_lock_waist_fitting +vis=True
```

This will create a folder `data/g1` with the optimized shape parameters in `data/g1/smpl_shape.pkl`. 

2. Retarget the AMASS dataset to the robot:

```bash
python fit_smpl_motion.py robot=g1_29dof_lock_waist_fitting +dataset_dir=/path/to/your/folder
```

This will output the retargeted motion in `data/g1/retargeted/retargeted_motion.pkl`. You can also specify the output file name by adding `+output_file_name=your_file_name`.

3. Visualize the retargeted motion:

```bash
python vis_retarget_data.py robot=g1_29dof_lock_waist_fitting
```

# Info about the retargeted data

Notice to distinguish the following terms:
- joint angles: the angles of the 1-Dof joints in the real robot model.
- joint positions: the cartesian positions of the joints (often not 1-Dof) in the SMPL model.
- pose_aa: the angle-axis representation of the joints (often not 1-Dof) in the SMPL model, which is a 3D vector for each joint.

The retargeted data is saved in a `pkl` file, which is a dictionary and can be loaded with `joblib.load`. Each key in the dictionary is a string representing the name of the motion, and the value is a dictionary containing the following keys:
- `root_pos`: the root position of the robot in the world frame, shape (N, 3).
- `root_rot`: the root quaternion (x, y, z, w) of the robot in the world frame, shape (N, 4).
- `pose_aa`: the angle-axis of each joint in the `Humanoid_Batch` model for the robot.
- `dof_pos`: the optimized joint angles of the robot that is best fit to the motion, shape (N, num_dof). It is in the order of the mujoco model. 
- `joint_pos_smpl`: the joint positions of the SMPL model, shape (N, num_joints, 3).

Please refer to [PHC](https://github.com/ZhengyiLuo/PHC) for more details.

