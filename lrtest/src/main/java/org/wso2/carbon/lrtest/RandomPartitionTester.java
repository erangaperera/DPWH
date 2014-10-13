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
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

/**
 * Main class for performing the random partition based model ensembler evaluation
 */
public class RandomPartitionTester {

	private static JavaSparkContext sc;

	@SuppressWarnings("serial")
	static class ParsePoint implements Function<String, LabeledPoint> {

		private static final Pattern COMMA = Pattern.compile(",");

		// Function for converting a csv line to a LabelPoint 
		public LabeledPoint call(String line) {
			Logger logger = Logger.getLogger(this.getClass());
			logger.debug(line);
			// System.out.println(line);
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
	 * @param fileLocation
	 * @param headerRowSkippingCriteria
	 * @return
	 */
	@SuppressWarnings("serial")
	private static JavaRDD<String> readData(String fileLocation,
			final String headerRowSkippingCriteria) {

		JavaRDD<String> lines = null;
		if (headerRowSkippingCriteria == null) {
			lines = sc.textFile(fileLocation);
		} else {
			lines = sc.textFile(fileLocation).filter(
					new Function<String, Boolean>() {
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
	 * Main method for performing the random partition based model ensembler evaluation
	 */
	public static void main(String[] args) {

		// Construction of Spark Configuration
		SparkConf sContext = new SparkConf();
		sContext.setMaster("local[4]");
		sContext.setAppName("JavaLR");
		sContext.set("spark.executor.memory", "4G");

		// Creates the spark context
		sc = new JavaSparkContext(sContext); // "local[4]", "JavaLR");

		// Load train and test data
		JavaRDD<String> trainingData = readData(
				"/Users/erangap/Documents/ML_Project/datasets/trainImputedNormalized.csv",
				"Id").sample(false, 0.1, 11L);
		JavaRDD<String> testdata = readData(
				"/Users/erangap/Documents/ML_Project/datasets/testImputedNormalized.csv",
				"Id").sample(false, 0.1, 11L);

		// trainingData.saveAsTextFile("/Users/erangap/Documents/ML_Project/datasets/reduced.csv");
		JavaRDD<LabeledPoint> points = trainingData.map(new ParsePoint());
		// points.persist(StorageLevel.MEMORY_AND_DISK());
		// System.out.println(points.first().features());
		JavaRDD<LabeledPoint> testPoints = testdata.map(new ParsePoint());
		// testPoints.persist(StorageLevel.MEMORY_AND_DISK());

		System.out.println("Total number of records -> " + points.count());

		RandomPartitionedEnSembler ensembler = new RandomPartitionedEnSembler();
		ensembler.setNoofModels(32);
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

		// Prepare data for testing
		JavaRDD<Double> testingLabels = testPoints.map(
				new Function<LabeledPoint, Double>() {

					private static final long serialVersionUID = -6597374940461185814L;

					public Double call(LabeledPoint dataPoint) throws Exception {
						return dataPoint.label();
					}
				}).cache();
		List<Double> classLabels = testingLabels.toArray();
		
		// Perform the predictions
		Date predictStartTime = Calendar.getInstance().getTime();
		String predictStart = dateFormat.format(predictStartTime);
		List<Double> predictedLabels = ensembler.voteAndPredit(testPoints).toArray();
		Date predictEndTime = Calendar.getInstance().getTime();
        String predictEnd = dateFormat.format(predictEndTime);
        
        // Predict time calculations and console print
        long preditElapsed = (predictEndTime.getTime() - predictStartTime.getTime()) / 1000;
        System.out.println("Prediction Started at -> " + predictStart);
		System.out.println("Prediction Ended at -> " + predictEnd);
		System.out.println("Time Taken to Predit -> " + preditElapsed + " Sec.");

		// Calculate and Display the accuracy
		System.out.println("Testing accuracy (%): "
				+ Metrics.accuracy(classLabels, predictedLabels));
		BinaryClassificationMetrics binaryClassificationMetrics = getBinaryClassificationMatrix(ensembler, testPoints);
		System.out.println("Area under the curve -> " + binaryClassificationMetrics.areaUnderROC());

	}
	
	/**
	 * This method calculates performance metrics for a given set of test scores and labels
	 * @param ensembler
	 * @param testingDataset
	 * @return
	 */
	private static BinaryClassificationMetrics getBinaryClassificationMatrix(final RandomPartitionedEnSembler ensembler, JavaRDD<LabeledPoint> testingDataset){
		JavaRDD<Tuple2<Object, Object>> scoreAndLabels = testingDataset.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
					
                	private static final long serialVersionUID = 8275673381396280119L;

					public Tuple2<Object, Object> call(LabeledPoint p) {
                        Double score = ensembler.voteAndPredit(p.features());
                        return new Tuple2<Object, Object>(score, p.label());
                    }
                }
        );
		return new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
	}

}
