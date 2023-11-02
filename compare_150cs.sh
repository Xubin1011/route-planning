#!/bin/bash

# 初始化变量来保存最小的charge time和计数charge time小于11.4h的文件数量
min_charge_time=9999999 # 初始化为一个大的值
count_charge_time_less_than_11_4=0

# 初始化变量来保存最小的travling time和计数travling time小于17.5h的文件数量
min_travling_time=9999999 # 初始化为一个大的值
count_travling_time_less_than_17_5=0

# 初始化变量来保存文件名
file_with_min_charge_time=""
file_with_min_travling_time=""

# 遍历文件
for ((i=1; i<=100; i++)); do
    file_name="/home/utlck/PycharmProjects/Tunning_results/compare/deploy_109_500epis_cs_150_$i.txt"

    # 使用 awk 从文件中提取 charge time 和 travling time 的值
    charge_time=$(tail -n 4 "$file_name" | awk -F'=' '/charge time/{print $2}' | sed 's/h//')
    travling_time=$(tail -n 4 "$file_name" | awk -F'=' '/travling time/{print $2}' | sed 's/h//')

    # 检查 charge time 是否小于最小值
    if (( $(awk -v charge_time="$charge_time" 'BEGIN {print (charge_time < 11.4) ? 1 : 0}') == 1 )); then
        count_charge_time_less_than_11_4=$((count_charge_time_less_than_11_4 + 1))
    fi

    # 检查 charge time 是否是最小值
    if (( $(awk -v charge_time="$charge_time" -v min_charge_time="$min_charge_time" 'BEGIN {print (charge_time < min_charge_time) ? 1 : 0}') == 1 )); then
        min_charge_time="$charge_time"
        file_with_min_charge_time="$file_name"
    fi

    # 检查 travling time 是否小于17.5小时
    if (( $(awk -v travling_time="$travling_time" 'BEGIN {print (travling_time < 17.5) ? 1 : 0}') == 1 )); then
        count_travling_time_less_than_17_5=$((count_travling_time_less_than_17_5 + 1))
    fi

    # 检查 travling time 是否是最小值
    if (( $(awk -v travling_time="$travling_time" -v min_travling_time="$min_travling_time" 'BEGIN {print (travling_time < min_travling_time) ? 1 : 0}') == 1 )); then
        min_travling_time="$travling_time"
        file_with_min_travling_time="$file_name"
    fi
done

# 输出charge time最小的文件名和小于11.4h的charge time文件数量
echo "File with the minimum charge time: $file_with_min_charge_time"
echo "Number of files with charge time less than 11.4h: $count_charge_time_less_than_11_4"

# 输出travling time最小的文件名和小于17.5h的travling time文件数量
echo "File with the minimum travling time: $file_with_min_travling_time"
echo "Number of files with travling time less than 17.5h: $count_travling_time_less_than_17_5"
