from subprocess import *
import time
call(['python.exe', 'ML_Project.py'])
time.sleep(2)
call(['python.exe', 'general_model.py'])
time.sleep(2)
call(['python.exe', 'models.py'])
time.sleep(2)
call(['python.exe', 'PCA1.py'])
time.sleep(2)
