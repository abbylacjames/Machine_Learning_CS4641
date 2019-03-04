
package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * 5000, 10000
 *
 * @author Hannah Lau
 * @author Aman Aggarwal
 * @version 1.0
 */
public class krkoptWeightTesting {
    private static Instance[] instances = initializeInstances();
    private static Instance[] testInstances = initializeTestInstances(); //TODO: created testing instances

    private static int inputLayer = 9, hiddenLayer = 10, outputLayer = 1, trainingIterations = 5000; //TODO: input layer = number of attributes, training iterations
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        int jk = 0;
        double RHCTestTotal = 0;
        double RHCTrainTotal = 0;
        double SATestTotal = 0;
        double SATrainTotal = 0;
        double GATestTotal = 0;
        double GATrainTotal = 0;
        for (int m = 0; m < 5; m++) {
            // System.out.println("IAM "+ m);
            for(int i = 0; i < oa.length; i++) {
                networks[i] = factory.createClassificationNetwork(
                                                                  new int[] {inputLayer, hiddenLayer, outputLayer});
                nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
            }

            oa[0] = new RandomizedHillClimbing(nnop[0]);
            oa[1] = new SimulatedAnnealing(1E11, .9, nnop[1]); //TODO:second parameter?
            oa[2] = new StandardGeneticAlgorithm(200, 100, 30, nnop[2]);

            for(int i = 0; i < oa.length; i++) {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                train(oa[i], networks[i], oaNames[i]); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10,9);

                Instance optimalInstance = oa[i].getOptimal();
                networks[i].setWeights(optimalInstance.getData());

                double predicted, actual;
                start = System.nanoTime();
                for(int j = 0; j < instances.length; j++) { //TODO: validation instances vs just instances
                    networks[i].setInputValues(instances[j].getData()); //validation.
                    networks[i].run();

                    predicted = Double.parseDouble(instances[j].getLabel().toString());
                    actual = Double.parseDouble(networks[i].getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10,9);

                results +=  "\nResults for training " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
                if (i == 1) {
                    RHCTrainTotal += correct/(correct+incorrect)*100;
                } else if (i == 2) {
                    SATrainTotal += correct/(correct+incorrect)*100 ;
                } else {
                    GATrainTotal += correct/(correct+incorrect)*100 ;
                }

                // double predicted, actual;
                incorrect = 0;
                correct = 0;
                start = System.nanoTime();
                for(int j = 0; j < testInstances.length; j++) { //TODO: validation instances vs just instances
                    networks[i].setInputValues(testInstances[j].getData()); //validation.
                    networks[i].run();

                    predicted = Double.parseDouble(testInstances[j].getLabel().toString());
                    actual = Double.parseDouble(networks[i].getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10,9);

                results +=  "\nResults for testing " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
                if (i == 1) {
                    RHCTestTotal += correct/(correct+incorrect)*100 ;
                } else if (i == 2) {
                    SATestTotal += correct/(correct+incorrect)*100 ;
                } else {
                    GATestTotal += correct/(correct+incorrect)*100 ;
                }
                // jk++;
                System.out.println(results);
                // System.out.println(jk);
            }
            // jk++;
            System.out.println(results);
            // System.out.println(jk);

        }
        double RHCTestAverage = RHCTestTotal/5;
        double RHCTrainAverage = RHCTrainTotal/5;
        double SATestAverage = SATestTotal/5;
        double SATrainAverage = SATrainTotal/5;
        double GATestAverage = GATestTotal/5;
        double GATrainAverage = GATrainTotal/5;
        System.out.println("RHC: train total " + RHCTrainTotal + " test total: " + RHCTestTotal);
        System.out.println("SA: train total " + SATrainTotal + " test total: " + SATestTotal);
        System.out.println("GA: train total " + GATrainTotal + " test total: " + GATestTotal);

        System.out.println("RHC: train average " + RHCTrainAverage + " test average: " + RHCTestAverage);
        System.out.println("SA: train average " + SATrainAverage + " test average: " + SATestAverage);
        System.out.println("GA: train average " + GATrainAverage + " test average: " + GATestAverage);



    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        // System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            // System.out.println(i+". : " + df.format(error));
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[614][][]; //TODO: change value in here?

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/csv_result-diabetes-train.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[9]; // 7 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 9; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }

    private static Instance[] initializeTestInstances() {

        double[][][] attributes = new double[144][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/csv_result-diabetes-test.csv")));
            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[9]; // change to number of attributes, and input layer
                attributes[i][1] = new double[1];

                for(int j = 0; j < 9; j++) //change here too
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }


}
