import os

# cmd_fedavg = 'docker compose -f .\FedAvg-0.5-MotionSense.yaml --compatibility up'
# os.system(cmd_fedavg)

# cmd_deev = 'docker compose -f DEEV-0.5-MotionSense.yaml --compatibility up'
# os.system(cmd_deev)

# cmd_deevper = 'docker compose -f DEEV-PER-MotionSense.yaml --compatibility up'
# os.system(cmd_deevper)

# cmd_deevpershared = 'docker compose -f DEEV-PER-SHARED-MotionSense.yaml --compatibility up'
# os.system(cmd_deevpershared)

# poc_cmd = 'docker compose -f POC-0.5-MotionSense.yaml --compatibility up'
# os.system(poc_cmd)

# cmd = 'docker compose -f DEEV-PER-SHARED-2-UCIHAR.yaml --compatibility up'
# os.system(cmd)

cmd = 'docker compose -f DEEV-PER-SHARED-2-MotionSense.yaml --compatibility up'
os.system(cmd)

cmd = 'docker compose -f DEEV-PER-SHARED-2-ExtraSensory.yaml --compatibility up'
os.system(cmd)

cmd = 'docker compose -f DEEV-PER-SHARED-3-UCIHAR.yaml --compatibility up'
os.system(cmd)

cmd = 'docker compose -f DEEV-PER-SHARED-3-MotionSense.yaml --compatibility up'
os.system(cmd)

cmd = 'docker compose -f DEEV-PER-SHARED-3-ExtraSensory.yaml --compatibility up'
os.system(cmd)

