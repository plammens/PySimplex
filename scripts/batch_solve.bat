@rem This batch file tests the main program with some sample

@del batch_solve.txt

@for /L %%n in (1,1,4) do ^
python scripts/solve.py 41 %%n --rule bland >> batch_solve.txt & ^
python scripts/solve.py 41 %%n --rule minrc >> batch_solve.txt

@for /L %%n in (1,1,4) do ^
python scripts/solve.py 70 %%n --rule bland >> batch_solve.txt & ^
python scripts/solve.py 70 %%n --rule minrc >> batch_solve.txt
