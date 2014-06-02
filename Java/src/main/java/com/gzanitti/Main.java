package com.gzanitti;

import com.gzanitti.linearregression.LinearRegression;
import com.gzanitti.logisticregression.LogisticRegression;
import org.ejml.simple.SimpleMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by gaston on 01/06/14.
 */
public class Main {
    /*
    args: nombre del archivo, largo del dataset, cantidad de features
     */

    public static void main(String[] args) {
        //DataSet data = new DataSet(args[0], args[1], args[2]);
        DataSet data = new DataSet("dataLinear", "4", "1");
        LinearRegression linear = new LinearRegression(data, 0.01);
        linear.train(200000);
        SimpleMatrix matrixLinear = new SimpleMatrix(2,1);
        matrixLinear.set(0, 0, 1.0);
        matrixLinear.set(1, 0, 6.0);
        System.out.println(linear.evaluate(matrixLinear));

        DataSet dataLogistic = new DataSet("dataLogistic", "4", "1");
        LogisticRegression logistic = new LogisticRegression(dataLogistic, 0.1);
        logistic.train(500000);
        SimpleMatrix matrixLogistic = new SimpleMatrix(2,1);
        matrixLogistic.set(0, 0, 1.0);
        matrixLogistic.set(1, 0, 6.0);
        System.out.println(logistic.evaluate(matrixLogistic));
    }
}
