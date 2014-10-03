package lrtest;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

public class LogisticRegresionTester {

	private static JavaSparkContext sc;

	@SuppressWarnings("serial")
	static class ParsePoint implements Function<String, LabeledPoint> {

		private static final Pattern COMMA = Pattern.compile(",");

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
			lines = sc.textFile(fileLocation).sample(false, 0.0001, 11L);
		} else {
			lines = sc.textFile(fileLocation).sample(false, 0.0001, 11L)
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

	public static void main(String[] args) {

		// SparkConf sContext = new SparkConf();
		// sContext.setMaster("local");
		// sContext.setAppName("JavaLR");
		// sContext.set("spark.executor.memory", "4G");
		
		Logger.getRootLogger().setLevel(Level.OFF);
		sc = new JavaSparkContext("local", "JavaLR");

		JavaRDD<String> data = readData(
				"/Users/erangap/Documents/ML_Project/datasets/trainImputedNormalized.csv",
				"Id");
		// JavaRDD<String> data =
		// readData("/Users/erangap/Documents/ML_Project/datasets/reduced.csv",
		// "Id");
		JavaRDD<String> trainingData = data;
		System.out.println(data.first());

		// trainingData.saveAsTextFile("/Users/erangap/Documents/ML_Project/datasets/reduced.csv");
		JavaRDD<LabeledPoint> points = trainingData.map(new ParsePoint())
				.cache();
		//long total = points.count();
//		JavaRDD<LabeledPoint> zeros = points
//				.filter(new Function<LabeledPoint, Boolean>() {
//
//					public Boolean call(LabeledPoint point) throws Exception {
//						// TODO Auto-generated method stub
//						return !(point.label() > 0);
//					}
//				});
//		long zerotot = zeros.count();
//		System.out
//				.println("Percentage of zeros ->" + (zerotot * 100.0 / total));
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		String start = dateFormat.format(Calendar.getInstance().getTime());
		final LogisticRegressionModel model = LogisticRegressionWithSGD.train(
				points.rdd(), 100);
		System.out.print("Final w: " + model.weights());
		// model.predict(points.first().features());
		System.out.println("");
//		final IntAdder ptot = new IntAdder();
//		final IntAdder pinc = new IntAdder();
//		final IntAdder pcor = new IntAdder();
		
		JavaRDD<LabeledPoint> testdata = points.sample(false, 0.001);
        JavaRDD<Vector> testingFeatures = testdata.map(new Function<LabeledPoint, Vector>() {
            public Vector call(LabeledPoint label) throws Exception {
                return label.features();
            }
        }).cache();

        JavaRDD<Double> testingLabels = testdata.map(new Function<LabeledPoint, Double>() {
            public Double call(LabeledPoint dataPoint) throws Exception {
                return dataPoint.label();
            }
        }).cache();
        
        List<Double> classLabels = testingLabels.toArray();
        List<Double> predictedLabels = model.predict(testingFeatures).toArray();
		
		
//		points.sample(false, 0.001).foreach(new VoidFunction<LabeledPoint>() {
//
//			public void call(LabeledPoint point) throws Exception {
//				// TODO Auto-generated method stub
//				double predicted = model.predict(point.features());
//				ptot.add();
//				if (point.label() == predicted)
//					pcor.add();
//				else
//					pinc.add();
//				
//			}
//		});
		System.out.println(start);
		System.out.println(dateFormat.format(Calendar.getInstance().getTime()));
//		System.out.println("Accuracy ->" + (pcor.getCount() * 100.0 / ptot.getCount()));
		System.out.println("Testing accuracy (%): " + Metrics.accuracy(classLabels, predictedLabels));
	}

}
