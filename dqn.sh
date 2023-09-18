#!/bin/bash
/home/utlck/.conda/envs/rp/bin/python /home/utlck/PycharmProjects/route-planning/dqn_n_pois.py &

python_pid=$!
start_time=$(date +%s)
echo -e "running dqn_n_pois.py"
echo -e "running dqn_n_pois.py"

while true; do
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    echo -e "\033[A\033[KElapsed time：$((elapsed_time / 60)) mins"
    if ! ps -p $python_pid > /dev/null; then
        echo -e "done"
        break
    fi
    
    sleep 60
done

read -p "Press [Enter] to exit.........."
