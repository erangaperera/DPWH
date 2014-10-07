package org.wso2.carbon.lrtest;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.wso2.carbon.lrtest.LogisticRegresionTester.ParsePoint;

public class HistogramTester {

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
		sContext.setMaster("local");
		sContext.setAppName("JavaLR2");
		sContext.set("spark.executor.memory", "4G");

	    Logger.getRootLogger().setLevel(Level.OFF);
		sc = new JavaSparkContext(sContext); // "local[4]", "JavaLR");
		JavaRDD<String> trainingData = readData(
				"/Users/erangap/Documents/ML_Project/datasets/trainImputedNormalized.csv",
				"Id").sample(false, 0.1, 11L);
		JavaRDD<String> testdata = //trainingData;
				readData(
				"/Users/erangap/Documents/ML_Project/datasets/testImputedNormalized.csv",
				"Id").sample(false, 0.1, 11L);

		JavaRDD<LabeledPoint> points = trainingData.map(new ParsePoint());
		System.out.println("Total number of records -> " + points.count());
		
		HistogramHelper histogramHelper = new HistogramHelper();
		histogramHelper.addDimension(0, 0.0, 1, 8);
		histogramHelper.addDimension(1, 0.0, 1, 8);
		histogramHelper.addDimension(2, 0.0, 1, 8);
		histogramHelper.addDimension(3, 0.0, 1, 8);
		histogramHelper.addDimension(4, 0.0, 1, 8);
		HistogramEnsembler ensembler = new HistogramEnsembler(histogramHelper, 32);
		ensembler.setThreshold(0.499999);
		
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		String trainStart = dateFormat.format(Calendar.getInstance().getTime());
		ensembler.train(points);
		String trainEnd = dateFormat.format(Calendar.getInstance().getTime());
		
		System.out.println("Training Started at -> " + trainStart);
		System.out.println("Training Ended at -> " + trainEnd);
		
		JavaRDD<LabeledPoint> testPoints = testdata.map(new ParsePoint());
		
		String predictStart = dateFormat.format(Calendar.getInstance().getTime());
		List<Double> predictedLabels = ensembler.predit(testPoints).toArray();
        String predictEnd = dateFormat.format(Calendar.getInstance().getTime());
        
        System.out.println("Prediction Started at -> " + predictStart);
		System.out.println("Prediction Ended at -> " + predictEnd);
		
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
