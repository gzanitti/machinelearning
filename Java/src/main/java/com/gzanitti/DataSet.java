package com.gzanitti;

import org.ejml.simple.SimpleMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by gaston on 01/06/14.
 */
public class DataSet {

    private SimpleMatrix mTrain;
    private SimpleMatrix mResult;
    private int mDataSetSize;
    private int mCantFeatures;

    public DataSet(String file, String rows, String columns) {
        mDataSetSize = Integer.parseInt(rows);
        mCantFeatures = Integer.parseInt(columns);

        mTrain = new SimpleMatrix(mDataSetSize, mCantFeatures + 1);
        mResult = new SimpleMatrix(mDataSetSize, 1);
        for (int i = 0; i < mDataSetSize; i++) {
            mTrain.set(i, 0, 1.0);
        }

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String sCurrentLine;
            for (int i = 0; i < mDataSetSize; i++) {
                sCurrentLine = br.readLine();
                String[] featuresAndResult = sCurrentLine.split(" ");
                for (int j = 0; j < featuresAndResult.length; j++) {
                    if(j < featuresAndResult.length - 1)
                        mTrain.set(i, j + 1, Double.parseDouble(featuresAndResult[j]));
                    else
                        mResult.set(i, 0, Double.parseDouble(featuresAndResult[j]));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int getDataSetSize() {
        return mDataSetSize;
    }
    
    public int getCantFeatures() {
        return mCantFeatures;
    }

    public SimpleMatrix getTrainPosition(int position) {
        SimpleMatrix vector = new SimpleMatrix(mCantFeatures + 1, 1);
        for (int i = 0; i < mCantFeatures + 1; i++) {
            vector.set(i, 0, mTrain.get(position, i));
        }
        return vector;
    }

    public Double getResultPosition(int position) {
        return mResult.get(position, 0);
    }

}
