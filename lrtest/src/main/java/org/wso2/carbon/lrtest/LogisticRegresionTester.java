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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.storage.StorageLevel;

public class LogisticRegresionTester {

	private static JavaSparkContext sc;
	private static final Log LOGGER = LogFactory.getLog(LogisticRegresionTester.class);

	@SuppressWarnings("serial")
	static class ParsePoint implements Function<String, LabeledPoint> {

		private static final Pattern COMMA = Pattern.compile(",");

		// Function for converting a csv line to a LabelPoint 
		public LabeledPoint call(String line) {
			LOGGER.debug(line);
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
						
						private static final long serialVersionUID = -4043460916941686047L;

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
	 * Main method for evaluating LogisticRegression Testing on entire dataset
	 * @param args
	 */
	public static void main(String[] args) {

		// Create spark configuration
		SparkConf sContext = new SparkConf();
		sContext.setMaster("local[4]");
		sContext.setAppName("JavaLR");
		sContext.set("spark.executor.memory", "4G");

		// Create Spark context
		sc = new JavaSparkContext(sContext); 
		
		// Load train and test data
		JavaRDD<String> trainingData = readData(
				"/Users/erangap/Documents/ML_Project/datasets/trainImputedNormalized.csv",
				"Id").sample(false, 0.1, 11L);
		JavaRDD<String> testdata = readData(
				"/Users/erangap/Documents/ML_Project/datasets/testImputedNormalized.csv",
				"Id").sample(false, 0.1, 11L);
		JavaRDD<LabeledPoint> points = trainingData.map(new ParsePoint());
		points.persist(StorageLevel.MEMORY_AND_DISK());
		JavaRDD<LabeledPoint> testPoints = testdata.map(new ParsePoint());
		testPoints.persist(StorageLevel.MEMORY_AND_DISK());
		System.out.println("Total number of records -> " + points.count());

		// Perform the training
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date trainStartTime = Calendar.getInstance().getTime();
		String trainStart = dateFormat.format(trainStartTime);
		LogisticRegressionModel model = LogisticRegressionWithSGD.train(
				points.rdd(), 10);
		Date trainEndTime = Calendar.getInstance().getTime();
		String trainEnd = dateFormat.format(trainEndTime);
		model.setThreshold(0.499999);
		
		// Training time calculations and console print
		long trainElapsed = (trainEndTime.getTime() - trainStartTime.getTime()) / 1000;
		System.out.println("Training Started at -> " + trainStart);
		System.out.println("Training Ended at -> " + trainEnd);
		System.out.println("Time Taken to Train -> " + trainElapsed + " Sec."); 

		// Prepare the Data for testing
		JavaRDD<Vector> testingFeatures = testPoints.map(
				new Function<LabeledPoint, Vector>() {
					public Vector call(LabeledPoint label) throws Exception {
						return label.features();
					}
				}).cache();

		JavaRDD<Double> testingLabels = testPoints.map(
				new Function<LabeledPoint, Double>() {
					public Double call(LabeledPoint dataPoint) throws Exception {
						return dataPoint.label();
					}
				}).cache();

		// Perform the prediction
		List<Double> classLabels = testingLabels.toArray();
		Date predictStartTime = Calendar.getInstance().getTime();
		String predictStart = dateFormat.format(predictStartTime);
		List<Double> predictedLabels = model.predict(testingFeatures).toArray();
		Date predictEndTime = Calendar.getInstance().getTime();
        String predictEnd = dateFormat.format(predictEndTime);

        // Predict time calculations and console print
        long preditElapsed = (predictEndTime.getTime() - predictStartTime.getTime()) / 1000;
        System.out.println("Prediction Started at -> " + predictStart);
		System.out.println("Prediction Ended at -> " + predictEnd);
		System.out.println("Time Taken to Predit -> " + preditElapsed + " Sec.");

		System.out.println("Testing accuracy (%): "
				+ Metrics.accuracy(classLabels, predictedLabels));
	}

}
