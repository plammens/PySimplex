del /f test.txt

for /L %%n in (1,1,4) do python solve.py 41 %%n bland >> batch_solve.txt & python solve.py 41 %%n foo >> batch_solve.txt

PAUSE