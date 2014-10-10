/*
 * Copyright (c) 2005-2014, WSO2 Inc. (http://www.wso2.org) All Rights Reserved.
 *
 * WSO2 Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.wso2.carbon.lrtest;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

/**
 * Main class for performing the analysis of Histogram based Partitioning
 */
public class HistogramTester {

	private static JavaSparkContext sc;
	private static Logger logger = Logger.getRootLogger();
	
	@SuppressWarnings("serial")
	static class ParsePoint implements Function<String, LabeledPoint> {

		private static final Pattern COMMA = Pattern.compile(",");
		
		// Function for converting a csv line to a LabelPoint 
		public LabeledPoint call(String line) {
			logger.debug(line);
			String[] parts = COMMA.split(line);
			double y = Double.parseDouble(parts[0]);
			double[] x = new double[parts.length - 1];
			for (int i = 1; i < parts.length; ++i) {
				x[i - 1] = Double.parseDouble(parts[i]);
			}
			return new LabeledPoint(y, Vectors.dense(x));
		}
	}

	/**
	 * This method will read a file into a JavaRDD
	 * 
	 * @param fileLocation
	 * @param headerRowSkippingCriteria
	 * @return JavaRDD
	 */
	private static JavaRDD<String> readData(String fileLocation,
			final String headerRowSkippingCriteria) {

		JavaRDD<String> lines = null;
		if (headerRowSkippingCriteria == null) {
			lines = sc.textFile(fileLocation);
		} else {
			lines = sc.textFile(fileLocation).filter(
					new Function<String, Boolean>() {

						private static final long serialVersionUID = 6988434865132848771L;

						public Boolean call(String line) {
							if (line.contains(headerRowSkippingCriteria)) {
								System.out.println(line);
								return false;
							} else
								return !(line
										.contains(headerRowSkippingCriteria));
						}
					});
		}
		return lines;
	}

	
	/**
	 * Main method to Perform Histogram Based Analysis 
	 * @param args
	 */
	public static void main(String[] args) {

		// Construction of Spark Configuration
		SparkConf sContext = new SparkConf();
		sContext.setMaster("local[4]");
		sContext.setAppName("JavaLR2");
		sContext.set("spark.executor.memory", "4G");

		// Create Spark context
		sc = new JavaSparkContext(sContext); // "local[4]", "JavaLR");
		Logger.getRootLogger().setLevel(Level.OFF);
		
		// Load train and test data
		JavaRDD<String> trainingData = readData(
				"/Users/erangap/Documents/ML_Project/datasets/trainImputedNormalized.csv",
				"Id").sample(false, 0.1, 11L);
		JavaRDD<String> testdata = //trainingData;
				readData(
				"/Users/erangap/Documents/ML_Project/datasets/testImputedNormalized.csv",
				"Id").sample(false, 0.1, 11L);

		JavaRDD<LabeledPoint> points = trainingData.map(new ParsePoint());
		System.out.println("Total number of records -> " + points.count());
		
		// Initialize the HistogramHelper
		HistogramHelper histogramHelper = new HistogramHelper();
		histogramHelper.addDimension(0, 0.0, 1, 10);
		histogramHelper.addDimension(1, 0.0, 1, 10);
		histogramHelper.addDimension(2, 0.0, 1, 10);
		histogramHelper.addDimension(3, 0.0, 1, 10);
		histogramHelper.addDimension(4, 0.0, 1, 10);
		// Initialize the HistogramEnsembler
		HistogramEnsembler ensembler = new HistogramEnsembler(histogramHelper, 32);
		// Default threshold should be change
		ensembler.setThreshold(0.499999);
		
		// Perform the training
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date trainStartTime = Calendar.getInstance().getTime();
		String trainStart = dateFormat.format(trainStartTime);
		ensembler.train(points);
		Date trainEndTime = Calendar.getInstance().getTime();
		String trainEnd = dateFormat.format(trainEndTime);

		// Training time calculations and console print
		long trainElapsed = (trainEndTime.getTime() - trainStartTime.getTime()) / 1000;
		System.out.println("Training Started at -> " + trainStart);
		System.out.println("Training Ended at -> " + trainEnd);
		System.out.println("Time Taken to Train -> " + trainElapsed + " Sec.");
		
		// Perform the predictions
		JavaRDD<LabeledPoint> testPoints = testdata.map(new ParsePoint());
		Date predictStartTime = Calendar.getInstance().getTime();
		String predictStart = dateFormat.format(predictStartTime);
		List<Double> predictedLabels = ensembler.predit(testPoints).toArray();
		Date predictEndTime = Calendar.getInstance().getTime();
        String predictEnd = dateFormat.format(predictEndTime);
        
        // Predict time calculations and console print
        long preditElapsed = (predictEndTime.getTime() - predictStartTime.getTime()) / 1000;
        System.out.println("Prediction Started at -> " + predictStart);
		System.out.println("Prediction Ended at -> " + predictEnd);
		System.out.println("Time Taken to Predit -> " + preditElapsed + " Sec.");
		
		// Calculate and Display the accuracy
		JavaRDD<Double> testingLabels = testPoints.map(new Function<LabeledPoint, Double>() {
            private static final long serialVersionUID = -3649842558466704526L;
			public Double call(LabeledPoint dataPoint) throws Exception {
                return dataPoint.label();
            }
        }).cache();
		List<Double> classLabels = testingLabels.toArray();
		System.out.println("Testing accuracy (%): " + Metrics.accuracy(classLabels, predictedLabels));

	}

}
