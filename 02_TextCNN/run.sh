rlaunch --cpu=8 --gpu=1 --memory=$((1024*100)) --preemptible=no --max-wait-time 5h -- python3 p7_TextCNN_train.py
