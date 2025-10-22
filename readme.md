根据locomo修改的车导航数据集生成程序

    python3 generate_sessions.py \
    --out-dir ./data \
    --prompt-dir ./prompt_examples \
    --events --session --summary --num-sessions 20 \
    --persona  \
    --num-days 120 --num-events 30 --max-turns-per-session 8 --num-events-per-session 1

python data_organizer.py --input driver.json --output Paul.json --speaker-a Paul --speaker-b CarBU-Agent

python qa_generator.py -i Paul.json -o Paul_qa.json -k 2

每次从driver_characters_all.json中选取一个“in_dataset=0”的人物，=