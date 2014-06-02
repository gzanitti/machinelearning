package com.gzanitti.logisticregression;

import com.gzanitti.DataSet;
import org.ejml.simple.SimpleMatrix;

/**
 * Created by gaston on 02/06/14.
 */
public class LogisticRegression {

    private DataSet mData;
    private SimpleMatrix mVectorTheta;
    private SimpleMatrix mVectorUpdate;
    private double mDelta;

    public LogisticRegression(DataSet data, double delta) {
        mData = data;
        mVectorTheta = new SimpleMatrix(1, mData.getCantFeatures() + 1);
        mVectorTheta.set(0.0);
        mDelta = delta;
    }

    public void train(double cantIter) {

        while(cantIter > 0) {

            SimpleMatrix acumVector = new SimpleMatrix(1, mData.getCantFeatures() + 1);
            acumVector.set(0.0);

            for (int i = 0; i <mData.getDataSetSize() ; i++) {
                SimpleMatrix trainPosition = mData.getTrainPosition(i);
                SimpleMatrix prod = mVectorTheta.mult(trainPosition);
                double sig = sigmoid(prod.get(0, 0));
                double dif = sig - mData.getResultPosition(i);

                for (int j = 0; j < mData.getCantFeatures() + 1; j++) {
                    double acum = dif * trainPosition.get(j, 0);
                    acumVector.set(0, j, acumVector.get(0, j) + acum);
                }
            }

            double scalar = mDelta * (1.0/mData.getDataSetSize());

            for (int j = 0; j < mData.getCantFeatures() + 1; j++) {
                double actualTheta = mVectorTheta.get(0, j);
                double newTheta = actualTheta - (scalar * acumVector.get(0, j));
                mVectorTheta.set(0, j, newTheta);
            }
            cantIter--;
        }
    }

    public double evaluate(SimpleMatrix valueVector) {
        double theta = mVectorTheta.mult(valueVector).get(0, 0);
        return sigmoid(theta);
    }

    public double sigmoid(double x) {
        double e = Math.pow(Math.E, -x);
        return (1.0/(1+e));
    }

}
