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

import java.util.regex.Pattern;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.wso2.carbon.lrtest.LogisticRegresionTester.ParsePoint;

public class GetMinMax {

	private static double[] min = new double[13];
	private static double[] max = new double[13];
	private static JavaSparkContext sc;
	private static final Pattern COMMA = Pattern.compile(",");

	public LabeledPoint call(String line) {
		Logger logger = Logger.getLogger(this.getClass());
		logger.debug(line);
		String[] parts = COMMA.split(line);
		double y = Double.parseDouble(parts[0]);
		double[] x = new double[parts.length - 1];
		for (int i = 1; i < parts.length; ++i) {
			x[i - 1] = Double.parseDouble(parts[i]);
		}
		return new LabeledPoint(y, Vectors.dense(x));
	}

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

	public static void main(String[] args) {

		SparkConf sContext = new SparkConf();
		sContext.setMaster("local[4]");
		sContext.setAppName("JavaLR2");
		sContext.set("spark.executor.memory", "4G");

		// Logger.getRootLogger().setLevel(Level.OFF);
		sc = new JavaSparkContext(sContext); // "local[4]", "JavaLR");
		JavaRDD<String> trainingData = readData(
				"/Users/erangap/Documents/ML_Project/datasets/trainImputedNormalized.csv",
				"Id"); // .sample(false, 0.01, 11L);
		for (int i = 0; i < min.length; i++) {
			min[i] = Double.POSITIVE_INFINITY;
			max[i] = Double.NEGATIVE_INFINITY;
		}
		JavaRDD<LabeledPoint> points = trainingData.map(new ParsePoint());
		points.foreach(new VoidFunction<LabeledPoint>() {

			private static final long serialVersionUID = -1174715752445463504L;

			public void call(LabeledPoint lbPoint) throws Exception {
				// System.out.println(lbPoint.label());
				double[] readval = lbPoint.features().toArray();
				for (int i = 0; i < 13; i++) {
					if (min[i] > readval[i])
						min[i] = readval[i];
					if (max[i] < readval[i])
						max[i] = readval[i];
				}
			}
		});
		for (int i = 0; i < min.length; i++) {
			System.out.println("Column " + i + " (Min,Max) ->(" + min[i] + ","
					+ max[i] + ")");
		}
	}

}
