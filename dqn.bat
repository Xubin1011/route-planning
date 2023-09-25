@echo off

set interpreter="E:\Program Files\miniconda3\envs\rp\python.exe"
set script="G:\OneDrive\Thesis\Code\route-planning\dqn_n_actions.py"
set try_number=38

echo Running dqn_n_actions.py......

set "start_time=!time!"

%interpreter% %script% %try_number%>> output_%try_number%.txt

set "end_time=!time!"

for /f "delims=:. tokens=1-4" %%a in ("!start_time!") do (
    set /a "start_seconds=(((%%a * 60) + %%b) * 60) + %%c"
)
for /f "delims=:. tokens=1-4" %%a in ("!end_time!") do (
    set /a "end_seconds=(((%%a * 60) + %%b) * 60) + %%c"
)

set /a "elapsed_seconds=end_seconds - start_seconds"
set /a "hours=elapsed_seconds / 3600"
set /a "minutes=(elapsed_seconds %% 3600) / 60"
set /a "seconds=elapsed_seconds %% 60"

echo Script completed in !hours!:!minutes!:!seconds!

echo done！！！

pause