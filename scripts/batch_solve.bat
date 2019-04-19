@comment
This batch file tests the main program with some sample
test cases.
@endcomment

@del batch_solve.txt

@for /L %%n in (1,1,4) do ^
    python solve.py 41 %%n --rule bland >> batch_solve.txt & ^
    python solve.py 41 %%n --rule minrc >> batch_solve.txt

@for /L %%n in (1,1,4) do ^
    python solve.py 70 %%n --rule bland >> batch_solve.txt & ^
    python solve.py 70 %%n --rule minrc >> batch_solve.txt

@pause
