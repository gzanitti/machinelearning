package com.gzanitti.linearregression;

import com.gzanitti.DataSet;
import org.ejml.simple.SimpleMatrix;

/**
 * Created by gaston on 01/06/14.
 */
public class LinearRegression {

    private DataSet mData;
    private SimpleMatrix mVectorTheta;
    private SimpleMatrix mVectorUpdate;
    private double mDelta;

    public LinearRegression(DataSet data, double delta) {
        mData = data;
        mVectorTheta = new SimpleMatrix(1, mData.getCantFeatures() + 1);
        mVectorTheta.set(0.0);
        mDelta = delta;
    }

    public void train(double iter) {

        while(iter > 0) {

            SimpleMatrix acumVector = new SimpleMatrix(1, mData.getCantFeatures() + 1);
            acumVector.set(0.0);

            for (int i = 0; i <mData.getDataSetSize() ; i++) {
                SimpleMatrix trainPosition = mData.getTrainPosition(i);
                SimpleMatrix prod = mVectorTheta.mult(trainPosition);
                double dif = prod.get(0, 0) - mData.getResultPosition(i);

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
            iter--;
        }
    }

    public double evaluate(SimpleMatrix valueVector) {
        return mVectorTheta.mult(valueVector).get(0, 0);
    }
}
