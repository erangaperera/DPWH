package org.wso2.carbon.lrtest;

import java.util.List;

public class Metrics {

    public static double accuracy(double[] actual, double[] predicted){

        assert (actual.length == predicted.length);

        double accuracy = 0.0;
        for(int i=0; i<actual.length;i++){
            if ( predicted[i] == actual[i]){
                accuracy++;
            }
        }

        return (accuracy/actual.length) * 100.0;
    }

    public static double accuracy(List<Double> actual, List<Double> predicted){

        assert (actual.size() == predicted.size());

        double accuracy = 0.0;
        for(int i=0; i<actual.size();i++){
            if ( predicted.get(i).doubleValue() == actual.get(i).doubleValue()){
                accuracy++;
            }
        }

        return (accuracy/actual.size()) * 100.0;
    }

}
