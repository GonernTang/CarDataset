根据locomo修改的车导航数据集生成程序

执行python3 generate_sessions.py \
    --out-dir ./data \
    --prompt-dir ./prompt_examples \
    --events --session --summary --num-sessions 20 \
    --persona  \
    --num-days 90 --num-events 40 --max-turns-per-session 8 --num-events-per-session 2

每次从driver_characters_all.json中选取一个“in_dataset=0”的人物，=