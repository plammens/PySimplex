del test.txt

for /L %%n in (1,1,4) do python solve.py 41 %%n bland >> batch_solve.txt & python solve.py 41 %%n no_bland >> batch_solve.txt

for /L %%n in (1,1,4) do python solve.py 70 %%n bland >> batch_solve.txt & python solve.py 70 %%n no_bland >> batch_solve.txt

PAUSE