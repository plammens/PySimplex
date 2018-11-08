del /f test.txt

for /L %%s in (1,1,70) do for /L %%n in (1,1,4) do python solve.py %%s %%n >> batch_solve.txt

PAUSE