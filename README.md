# PolLeakageMetric

leakage_bound.py can be run from the command line, or you can import the class LeakageMetric and use it in your own program.

From the command line:

Assuming a bunch of CST text files are in a directory /home/user/CST_output/:

> python leakage_bound.py -N none -o output_file.txt -f /home/user/CST_output/*.txt

will compute the leakage bound with no normalization as a function of frequency and save the result to the file output_file.txt.

It also has a quick plot mode, with either linear scale (--plot-lin) or log scale (--plot-log): 
> python leakage_bound.py -N none --plot-log -f /home/user/CST_output/*.txt

Use of the object is demonstrated in example.py
