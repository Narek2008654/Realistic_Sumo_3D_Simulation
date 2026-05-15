@echo off
call C:\Users\User\miniforge3\Scripts\activate.bat sumo
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUNBUFFERED=1
cd /d c:\Users\User\Robotics\Realistic_Sumo_3D_Simulation
python -u train_dqn_3d.py
