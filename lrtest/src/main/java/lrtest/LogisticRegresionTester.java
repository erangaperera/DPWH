package lrtest;

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
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.optimization.LBFGS;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.storage.StorageLevel;

public class LogisticRegresionTester {

	private static JavaSparkContext sc;

	@SuppressWarnings("serial")
	static class ParsePoint implements Function<String, LabeledPoint> {

		private static final Pattern COMMA = Pattern.compile(",");

//		// for abc.txt
//		public LabeledPoint call(String line) {
//			Logger logger = Logger.getLogger(this.getClass());
//			logger.debug(line);
//			// System.out.println(line);
//			String[] parts = COMMA.split(line);
//			double y = Double.parseDouble(parts[parts.length - 1]);
//			double[] x = new double[parts.length - 1];
//			for (int i = 0; i < parts.length - 1; ++i) {
//				x[i] = Double.parseDouble(parts[i]);
//			}
//			return new LabeledPoint(y, Vectors.dense(x));
//		}
		
		// for Upuls dataset
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

	@SuppressWarnings("serial")
	private static JavaRDD<String> readData(String fileLocation,
			final String headerRowSkippingCriteria) {

		JavaRDD<String> lines = null;
		if (headerRowSkippingCriteria == null) {
			lines = sc.textFile(fileLocation).sample(false, 0.02, 11L);
		} else {
			lines = sc.textFile(fileLocation).sample(false, 0.02, 11L)
					.filter(new Function<String, Boolean>() {
						public Boolean call(String line) {
							if (line.contains(headerRowSkippingCriteria)) {
								System.out.println(line);
								return false;
							} else
								return !(line.contains(headerRowSkippingCriteria));
						}
					});
		}
		return lines;
	}

	@SuppressWarnings("serial")
	public static void main(String[] args) {

		 SparkConf sContext = new SparkConf();
		 sContext.setMaster("local[4]");
		 sContext.setAppName("JavaLR");
		 sContext.set("spark.executor.memory", "4G");
		
		Logger.getRootLogger().setLevel(Level.OFF);
		sc = new JavaSparkContext(sContext); //"local[4]", "JavaLR");
		JavaRDD<String> trainingData = readData(
				//"/Users/erangap/Documents/ML_Project/abc.txt",
				"/Users/erangap/Documents/ML_Project/datasets/trainImputedNormalized.csv",
				"Id");
//		JavaRDD<String> trainingData = data;
		JavaRDD<String> testdata = readData(
				//"/Users/erangap/Documents/ML_Project/abc.txt",
				"/Users/erangap/Documents/ML_Project/datasets/testImputedNormalized.csv",
				"Id");
		// System.out.println(data.first());

		// trainingData.saveAsTextFile("/Users/erangap/Documents/ML_Project/datasets/reduced.csv");
		JavaRDD<LabeledPoint> points = trainingData.map(new ParsePoint());
		points.persist(StorageLevel.MEMORY_AND_DISK());
		System.out.println(points.first().features());
		JavaRDD<LabeledPoint> testPoints = testdata.map(new ParsePoint());
		testPoints.persist(StorageLevel.MEMORY_AND_DISK());
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		String start = dateFormat.format(Calendar.getInstance().getTime());
		LogisticRegressionModel model = 
				LogisticRegressionWithSGD.train(points.rdd(), 100);
		// model.clearThreshold();
		model.setThreshold(0.499999);
		System.out.print("Final w: " + model.weights());
//		model.weights().toBreeze().
		System.out.println("");
		
//		JavaRDD<LabeledPoint> testdata = points.sample(false, 0.1);
        JavaRDD<Vector> testingFeatures = testPoints.map(new Function<LabeledPoint, Vector>() {
            public Vector call(LabeledPoint label) throws Exception {
                return label.features();
            }
        }).cache();

        JavaRDD<Double> testingLabels = testPoints.map(new Function<LabeledPoint, Double>() {
            public Double call(LabeledPoint dataPoint) throws Exception {
                return dataPoint.label();
            }
        }).cache();
        
        List<Double> classLabels = testingLabels.toArray();
        List<Double> predictedLabels = model.predict(testingFeatures).toArray();
		
		
		System.out.println(start);
		System.out.println(dateFormat.format(Calendar.getInstance().getTime()));
		for (Double predlabel : predictedLabels) {
			if (predlabel > 0)
				System.out.println("Predited 1");
		}
		System.out.println("Testing accuracy (%): " + Metrics.accuracy(classLabels, predictedLabels));
	}

}
