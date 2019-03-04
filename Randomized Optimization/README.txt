Abigail James
ajames45

I used ABAGAIL for the entire project and some code was borrowed from Aman Aggarwal. 
Changes were made to fit my own data and run my own tests

Part 1:
krkoptWeightedTests.java runs the results for optimizing neural networks
copy this file and the two csv files to this directory ABAGAIL-master->src->opt->test
results_diabetes-diabestes-train.csv and results_diabetes-diabestes-test.csv 

Run it by executing these commands
cd into directory
ant
java -cp ABAGAIL.jar opt.test.krkoptWeightTesting

When this is run, it will print out the results for each run before the average is taken 
5000 iterations will run and print out the data
Part 2:
I completed the CountingOnes,FourPeaks, and TwoColor Tests 
When these are run it will print out the results. 
To test these, I made for loops and tried differnet parameters. This code is no longer there 

To run:
cd into directory
ant
java -cp ABAGAIL.jar opt.test.CountOnesTest
java -cp ABAGAIL.jar opt.test.FourPeaksTest
java -cp ABAGAIL.jar opt.test.TwoColorsTest