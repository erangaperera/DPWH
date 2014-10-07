package org.wso2.carbon.lrtest;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.mllib.linalg.Vector;

import scala.Serializable;
import scala.Tuple2;

public class RandomPartitionedEnSembler implements Serializable{

	private static final long serialVersionUID = -3666105986831973419L;

	private final class PartionFilterFunction implements
			Function<Tuple2<Integer, LabeledPoint>, Boolean> {
		
		private static final long serialVersionUID = 2895249051284874295L;
		private int filter;
		public PartionFilterFunction(int filter) {
			this.filter = filter;
		}
		public Boolean call(Tuple2<Integer, LabeledPoint> in) throws Exception {
			return (in._1.intValue() == filter);
		}
	}

	private List<LogisticRegressionModel> models = new ArrayList<LogisticRegressionModel>();
	private int noofModels = 0;
	private double threshold = 0.5;
	
	public void setNoofModels(int noofModels) {
		this.noofModels = noofModels;
	}
	
	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	public void train(JavaRDD<LabeledPoint> data){
		
		JavaPairRDD<Integer, LabeledPoint> points = data.flatMapToPair(new PairFlatMapFunction<LabeledPoint, Integer, LabeledPoint>() {

			private static final long serialVersionUID = 3787644685293675220L;

			public Iterable<Tuple2<Integer, LabeledPoint>> call(LabeledPoint labelPoint)
					throws Exception {
				List<Tuple2<Integer, LabeledPoint>> list = new ArrayList<Tuple2<Integer,LabeledPoint>>();
				list.add(new Tuple2<Integer, LabeledPoint>((int)(Math.random() * noofModels), labelPoint));
				return list;
			}
		});
		points.persist(StorageLevel.MEMORY_AND_DISK());
		
		for (int i = 0; i < noofModels; i++) {
			JavaPairRDD<Integer, LabeledPoint> modeldata = points.filter(new PartionFilterFunction(i));
			LogisticRegressionModel model = LogisticRegressionWithSGD.train(modeldata.values().rdd(), 100); 
			model.setThreshold(threshold);
			models.add(model);
		}
	}
	
	public double voteAndPredit(Vector v){
		double zero = 0 ,one = 0;
		// Vote the prediction
		for (int i = 0; i < noofModels; i++) {
			double predict = models.get(i).predict(v);
			if (predict == 0) {
				zero++;
			}
			else {
				one++;
			}
		}
		// Evaluate the voting
		if (one > zero) {
			return 1.0;
		}
		else {
			return 0.0;
		}
	}
	
	public JavaRDD<Double> voteAndPredit(JavaRDD<LabeledPoint> labelpoints){
		return labelpoints.map(new Function<LabeledPoint, Double>() {
			private static final long serialVersionUID = -4629831955936964163L;
			public Double call(LabeledPoint labelpoint) throws Exception {
				return voteAndPredit(labelpoint.features());
			}
		});
	}
}
